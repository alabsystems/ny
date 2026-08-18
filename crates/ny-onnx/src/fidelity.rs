// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph-fidelity gate: does NY's loaded network still have the authored
//! ONNX weights, bit for bit?
//!
//! NY's loader rewrites the graph before any bound is propagated. The largest
//! rewrite is [`crate::loader`]'s Conv/Gemm + `BatchNormalization` fold, which
//! multiplies the authored kernel by the BN scale **in f32** and writes the
//! product back under the authored tensor name. Every certificate NY emits is
//! therefore a statement about the *post-load* function. When the post-load
//! weights differ from the authored initializers, that function is not the
//! benchmark network, and a certificate must say so.
//!
//! This module is a **diagnostic only**. It does not change loading, bound
//! propagation, or any verdict. It reports, per authored node and per weight
//! tensor:
//!
//! 1. **vs authored** — the bit-diff against the raw ONNX payload, taken from
//!    both places ONNX authors constants (graph initializers and `Constant` node
//!    attributes): max absolute deviation, max relative deviation, max ULP
//!    distance, and the number of differing elements. A model where every tensor
//!    is [`TensorStatus::Identical`] and nothing was synthesized or dropped is
//!    the authored graph ([`GraphFidelityReport::is_authored_graph`]).
//! 2. **vs f64 fold reference** — for a rewrite explained by a BN fold, the
//!    same expression re-evaluated in f64 from the authored initializers. This
//!    separates "the rewrite is algebraically intended" from "the rewrite lost
//!    precision": the deviation here is the f32 rounding NY's certified
//!    enclosure does *not* account for.
//!
//! The f64 reference is deliberately the same expression tree the loader uses
//! (`scale = gamma / sqrt(var + eps)`, `shift = beta - gamma * mean / sqrt(var + eps)`,
//! `W' = W * scale`, `b' = b * scale + shift`), evaluated with exactly-widened
//! f32 inputs. It is not a claim about what the fold *should* be; it isolates
//! the precision of NY's evaluation of it.
//!
//! Attribution only ever *explains an observed rewrite*. A tensor that survives
//! bit-identically is never given a fold reference, so the loader's many
//! fold-declining guards (weight tying, non-default Gemm affine, grouped
//! `ConvTranspose`, externally observable values) cannot produce a spurious
//! deviation.

use crate::onnx_proto::{self, attribute_type};
use crate::{load_onnx_bytes, WeightStore};
use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use prost::Message;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// `TensorProto.DataType.FLOAT`. Only FLOAT initializers can be bit-compared:
/// `WeightStore` normalizes other authored dtypes to f32 on load, so a
/// post-load difference would not distinguish a rewrite from that widening.
const ONNX_TENSOR_FLOAT32: i32 = 1;
/// `TensorProto.DataLocation.EXTERNAL`.
const ONNX_DATA_LOCATION_EXTERNAL: i32 = 1;
/// Loader default when a `BatchNormalization` node omits `epsilon`; kept as f32
/// so the f64 reference widens the same value the loader used.
const DEFAULT_BATCH_NORM_EPSILON: f32 = 1.0e-5;

/// Deviation of a post-load tensor from a reference tensor of the same shape.
///
/// `max_rel` is taken over elements with a nonzero reference only; elements
/// whose reference is zero contribute to `max_abs`, `max_ulp` and
/// `elements_differing` but cannot have a relative deviation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Deviation {
    /// Element count compared.
    pub elements: usize,
    /// Elements whose post-load bits differ from the reference bits.
    pub elements_differing: usize,
    /// Largest `|post_load - reference|`.
    pub max_abs: f64,
    /// Largest `|post_load - reference| / |reference|` over nonzero references.
    pub max_rel: f64,
    /// Largest f32 ULP distance between the post-load value and the reference
    /// rounded to f32.
    pub max_ulp: u32,
}

impl Deviation {
    /// Whether every compared element matched the reference bit for bit.
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.elements_differing == 0
    }

    fn worst(self, other: Self) -> Self {
        Self {
            elements: self.elements + other.elements,
            elements_differing: self.elements_differing + other.elements_differing,
            max_abs: nan_aware_max(self.max_abs, other.max_abs),
            max_rel: nan_aware_max(self.max_rel, other.max_rel),
            max_ulp: self.max_ulp.max(other.max_ulp),
        }
    }

    /// Deviation of `current` from an f64 `reference` of identical shape.
    fn measure(current: &ArrayD<f32>, reference: &ArrayD<f64>) -> Option<Self> {
        if current.shape() != reference.shape() {
            return None;
        }
        let mut out = Self {
            elements: current.len(),
            ..Self::default()
        };
        for (&value, &want) in current.iter().zip(reference.iter()) {
            let rounded = want as f32;
            let ulp = ulp_distance(value, rounded);
            if ulp != 0 {
                out.elements_differing += 1;
            }
            out.max_ulp = out.max_ulp.max(ulp);
            let abs = (f64::from(value) - want).abs();
            out.max_abs = nan_aware_max(out.max_abs, abs);
            if want != 0.0 {
                out.max_rel = nan_aware_max(out.max_rel, abs / want.abs());
            }
        }
        Some(out)
    }
}

/// A NaN-propagating maximum: a NaN deviation must not be silently discarded.
fn nan_aware_max(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        return f64::NAN;
    }
    if right > left {
        right
    } else {
        left
    }
}

/// Monotonic ordering key for f32 bits, so ULP distance is a plain subtraction.
fn ulp_key(value: f32) -> u32 {
    // Canonicalize -0.0 to +0.0: the two are numerically equal and must not
    // report a 1-ULP rewrite.
    let value = if value == 0.0 { 0.0 } else { value };
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

/// f32 ULP distance; `u32::MAX` when either side is NaN.
fn ulp_distance(left: f32, right: f32) -> u32 {
    if left.is_nan() || right.is_nan() {
        return u32::MAX;
    }
    ulp_key(left).abs_diff(ulp_key(right))
}

/// How a post-load weight tensor relates to the authored ONNX initializer of
/// the same name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorStatus {
    /// Authored initializer survived load bit for bit.
    Identical,
    /// Same shape, at least one differing bit: a load-time rewrite.
    Rewritten,
    /// A load-time rewrite changed the tensor's shape.
    Reshaped,
    /// Authored initializer is no longer in the post-load weight store.
    Dropped,
    /// Post-load weight with no authored initializer of that name.
    Synthesized,
    /// Authored payload could not be read for comparison (external data, or a
    /// non-FLOAT authored dtype that `WeightStore` normalizes on load).
    Undetermined,
}

impl TensorStatus {
    /// Whether this status is compatible with "NY loaded the authored graph".
    #[must_use]
    pub fn is_faithful(self) -> bool {
        matches!(self, Self::Identical)
    }

    /// Lowercase label for text output.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Identical => "identical",
            Self::Rewritten => "rewritten",
            Self::Reshaped => "reshaped",
            Self::Dropped => "dropped",
            Self::Synthesized => "synthesized",
            Self::Undetermined => "undetermined",
        }
    }
}

/// The load-time rewrite a tensor's deviation is attributed to.
#[derive(Debug, Clone, Serialize)]
pub struct FoldAttribution {
    /// Authored `BatchNormalization` node folded away.
    pub batch_norm_node: String,
    /// Authored node that absorbed the BN affine.
    pub host_node: String,
    /// Which loader fold pattern this is: `conv/gemm+bn`,
    /// `gemm->reshape->bn`, or `bn->reshape->gemm`.
    pub pattern: String,
    /// Which fold product this tensor is (`weight` or `bias`).
    pub role: String,
    /// Deviation of NY's f32 fold from the same expression evaluated in f64.
    pub deviation: Deviation,
}

/// Where the authored value of a tensor came from, and how it was compared.
///
/// ONNX authors constants in two places — graph initializers and `Constant`
/// node attributes — and NY's loader lifts both into one name-keyed store. A
/// diagnostic that only read initializers would report every lifted `Constant`
/// as a synthesized weight, which is a false alarm: the value *is* authored.
/// Integer payloads (`Reshape` shape vectors, `Gather` indices) are compared as
/// integers, because the store also keeps a lossy f32 mirror of them that is not
/// a rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorKind {
    /// FLOAT graph initializer, compared bit for bit.
    FloatInitializer,
    /// FLOAT `Constant` node value, compared bit for bit.
    FloatConstant,
    /// Integer graph initializer, compared exactly as integers.
    IntegerInitializer,
    /// Integer `Constant` node value, compared exactly as integers.
    IntegerConstant,
    /// Post-load tensor with no authored payload anywhere in the ONNX.
    Synthesized,
    /// Authored, but its payload is not readable from this file.
    Unreadable,
}

impl TensorKind {
    /// Whether this tensor carries float network coefficients.
    #[must_use]
    pub fn is_float_weight(self) -> bool {
        matches!(self, Self::FloatInitializer | Self::FloatConstant)
    }

    /// Lowercase label for text output.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::FloatInitializer => "f32-init",
            Self::FloatConstant => "f32-const",
            Self::IntegerInitializer => "int-init",
            Self::IntegerConstant => "int-const",
            Self::Synthesized => "synthesized",
            Self::Unreadable => "unreadable",
        }
    }
}

/// Per-tensor fidelity record.
#[derive(Debug, Clone, Serialize)]
pub struct TensorFidelity {
    /// Weight tensor name (authored, or synthesized by a rewrite).
    pub name: String,
    /// Where the authored value came from and how it was compared.
    pub kind: TensorKind,
    /// Status against the authored initializer of the same name.
    pub status: TensorStatus,
    /// Authored shape, when an authored initializer existed and was readable.
    pub authored_shape: Option<Vec<usize>>,
    /// Post-load shape, when the tensor is in the post-load store.
    pub post_load_shape: Option<Vec<usize>>,
    /// Whether the post-load store keeps an integer copy of this tensor. An
    /// integer tensor is structural (a `Reshape` shape, a `Gather` index), not a
    /// float coefficient of the network.
    pub post_load_integer: bool,
    /// Authored graph nodes that consume this tensor, as `op_type:node`.
    pub consumers: Vec<String>,
    /// Deviation against the authored initializer, when both sides exist with
    /// the same shape.
    pub vs_authored: Option<Deviation>,
    /// Deviation against the f64 reference for the rewrite that produced it.
    pub fold: Option<FoldAttribution>,
    /// Why the authored payload could not be compared, when applicable.
    pub note: Option<String>,
}

/// Per-node roll-up of the fidelity of the weight tensors a node consumes.
///
/// A roll-up, not a source of truth. Every deviation here is a maximum over
/// [`GraphFidelityReport::tensors`] entries, and the mapping from a synthesized
/// tensor back to the node that gained it is found by matching the node's weight
/// input against NY's post-load layers. That match is exact for every fold the
/// loader performs (each requires its weight to have a single consumer), but
/// weight-tied graphs could in principle charge a synthesized bias to the wrong
/// node. The per-tensor table is always exact; read it when attribution matters.
#[derive(Debug, Clone, Serialize)]
pub struct NodeFidelity {
    /// Authored node name, falling back to its first output when unnamed —
    /// the same identity the loader gives the resulting layer.
    pub node: String,
    /// Authored ONNX `op_type`.
    pub op_type: String,
    /// Whether a post-load layer still carries this node's identity or weights.
    pub present_post_load: bool,
    /// Weight tensor names attributed to this node, authored then synthesized.
    pub tensors: Vec<String>,
    /// Worst status over this node's tensors, by severity.
    pub status: TensorStatus,
    /// Worst deviation vs the authored initializers over this node's tensors.
    pub vs_authored: Deviation,
    /// Worst deviation vs the f64 fold reference over this node's tensors.
    pub vs_fold_reference: Option<Deviation>,
}

impl NodeFidelity {
    /// Whether this node survives load with the authored weights, bit for bit.
    ///
    /// A node the loader removed is not faithful even when the tensors it used
    /// to consume are still in the weight store untouched: after a BN fold the
    /// four BN statistics stay in the store bit-identically, orphaned, while
    /// the node that read them is gone.
    ///
    /// `Constant` is the one exception. The loader always lifts a `Constant`
    /// node's value into the weight store and drops the node; that is a
    /// representation change, not a rewrite, and its faithfulness is decided by
    /// whether the lifted value still matches ([`Self::status`]).
    #[must_use]
    pub fn is_faithful(&self) -> bool {
        (self.present_post_load || self.is_lifted_constant())
            && (self.status.is_faithful() || self.tensors.is_empty())
    }

    /// Whether this is a `Constant` node the loader lifted into the store.
    #[must_use]
    pub fn is_lifted_constant(&self) -> bool {
        self.op_type == "Constant"
    }

    /// Status label for text output, surfacing node removal.
    #[must_use]
    pub fn label(&self) -> &'static str {
        if self.present_post_load {
            self.status.label()
        } else if self.is_lifted_constant() {
            "lifted"
        } else {
            "node-removed"
        }
    }
}

/// Result of bit-diffing NY's post-load weights against the authored ONNX
/// initializers.
#[derive(Debug, Clone, Serialize)]
pub struct GraphFidelityReport {
    /// Model identity as supplied by the caller.
    pub model: String,
    /// Authored initializers in the ONNX graph.
    pub authored_initializers: usize,
    /// Authored `Constant` node values the loader lifts into the weight store.
    pub authored_constants: usize,
    /// Authored payloads that could be read for comparison.
    pub authored_comparable: usize,
    /// Authored payloads that could not be read: external data, or a dtype this
    /// tool does not decode.
    pub authored_undetermined: usize,
    /// Tensors in NY's post-load weight store.
    pub post_load_weights: usize,
    /// Authored `BatchNormalization` nodes in the ONNX graph.
    pub authored_batch_norm_nodes: usize,
    /// `BatchNorm` layers surviving in NY's post-load network.
    pub post_load_batch_norm_layers: usize,
    /// Per-tensor records, authored graph order first, then synthesized.
    pub tensors: Vec<TensorFidelity>,
    /// Per-node roll-up in authored graph order.
    pub nodes: Vec<NodeFidelity>,
}

impl GraphFidelityReport {
    /// Count of tensors with the given status.
    #[must_use]
    pub fn count(&self, status: TensorStatus) -> usize {
        self.tensors
            .iter()
            .filter(|tensor| tensor.status == status)
            .count()
    }

    /// Whether NY's post-load weights are exactly the authored ones.
    ///
    /// True only when every authored payload — graph initializer or `Constant`
    /// node value — was readable and survived load unchanged (bit for bit for
    /// FLOAT, exactly for integer), nothing was dropped or reshaped, and no
    /// weight was synthesized. When this is false, a certificate about the
    /// loaded network is a certificate about a rewrite, not about the benchmark
    /// network.
    ///
    /// This is a claim about weight VALUES only. It does not assert the graph
    /// topology survived: read [`Self::removed_nodes`] alongside it.
    #[must_use]
    pub fn is_authored_graph(&self) -> bool {
        self.tensors
            .iter()
            .all(|tensor| tensor.status.is_faithful())
    }

    /// Whether every FLOAT coefficient of the network is the authored one.
    ///
    /// Weaker than [`Self::is_authored_graph`] in exactly one way: it forgives a
    /// synthesized tensor whose post-load payload is an INTEGER. NY's
    /// constant-folder materializes shape arithmetic
    /// (`Shape -> Gather -> Unsqueeze -> Concat -> Reshape`) into new integer
    /// store entries — measured on `dist_shift_2023/mnist_prior.onnx`, which
    /// gains four such tensors while all eight of its f32 weights stay
    /// bit-identical. That changes the graph's representation, not the real
    /// function being certified.
    ///
    /// This is the predicate a certificate's scope claim should quote. When it
    /// is false, the certified function is not the benchmark network.
    #[must_use]
    pub fn float_weights_are_authored(&self) -> bool {
        self.tensors.iter().all(|tensor| match tensor.status {
            TensorStatus::Identical => true,
            TensorStatus::Synthesized => tensor.post_load_integer,
            _ => false,
        })
    }

    /// Synthesized integer tensors: constant-folded shape arithmetic, which
    /// [`Self::float_weights_are_authored`] forgives and
    /// [`Self::is_authored_graph`] does not.
    #[must_use]
    pub fn structural_additions(&self) -> usize {
        self.tensors
            .iter()
            .filter(|tensor| tensor.status == TensorStatus::Synthesized && tensor.post_load_integer)
            .count()
    }

    /// Authored nodes with no surviving post-load layer or weight consumer.
    ///
    /// Lifted `Constant` nodes are excluded: the loader always turns those into
    /// store entries, which is a representation change rather than a rewrite of
    /// the graph's computation.
    #[must_use]
    pub fn removed_nodes(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| !node.present_post_load && !node.is_lifted_constant())
            .count()
    }

    /// Rewritten or synthesized tensors with no fold attribution — a rewrite
    /// this diagnostic cannot explain, and the loudest signal in the report.
    #[must_use]
    pub fn unexplained_rewrites(&self) -> Vec<&TensorFidelity> {
        self.tensors
            .iter()
            .filter(|tensor| {
                matches!(
                    tensor.status,
                    TensorStatus::Rewritten | TensorStatus::Reshaped | TensorStatus::Synthesized
                ) && tensor.fold.is_none()
            })
            .collect()
    }

    /// Worst deviation of any rewritten tensor from the authored initializer.
    #[must_use]
    pub fn worst_vs_authored(&self) -> Deviation {
        self.tensors
            .iter()
            .filter(|tensor| !tensor.status.is_faithful())
            .filter_map(|tensor| tensor.vs_authored)
            .fold(Deviation::default(), Deviation::worst)
    }

    /// Worst deviation of NY's f32 fold products from the f64 reference fold.
    /// This is the precision the certified enclosure does not account for.
    #[must_use]
    pub fn worst_vs_fold_reference(&self) -> Deviation {
        self.tensors
            .iter()
            .filter_map(|tensor| tensor.fold.as_ref())
            .map(|fold| fold.deviation)
            .fold(Deviation::default(), Deviation::worst)
    }

    /// Worst fold-reference deviation restricted to one fold product, `weight`
    /// or `bias`.
    ///
    /// Worth reading separately. A folded weight is one f32 multiply, so it
    /// stays within a couple of ULP. A folded bias carries
    /// `beta - gamma * mean / sqrt(var + eps)` — a subtraction that can cancel —
    /// and, for `bn->reshape->gemm`, an f32 accumulation over every input
    /// feature. The bias term is where the rewrite's unaccounted error lives.
    #[must_use]
    pub fn worst_fold_deviation_for_role(&self, role: &str) -> Deviation {
        self.tensors
            .iter()
            .filter_map(|tensor| tensor.fold.as_ref())
            .filter(|fold| fold.role == role)
            .map(|fold| fold.deviation)
            .fold(Deviation::default(), Deviation::worst)
    }
}

/// Bit-diff NY's post-load weights for `path` against its authored ONNX
/// initializers.
///
/// Reads the model twice: once through NY's full loader (all fusions active,
/// exactly as verification sees it) and once as a raw protobuf.
pub fn graph_fidelity_report<P: AsRef<Path>>(path: P) -> Result<GraphFidelityReport> {
    let path = path.as_ref();
    let bytes = ny_load::io::read_bytes_maybe_gzip(path)?;
    let name = path.to_string_lossy().into_owned();
    graph_fidelity_report_bytes(&name, &bytes)
}

/// Bit-diff NY's post-load weights against authored initializers, for an
/// in-memory ONNX payload. `name` is used for the loader's model identity and
/// for reporting only.
///
/// External-data initializers are reported [`TensorStatus::Undetermined`]:
/// their authored payload lives outside these bytes.
pub fn graph_fidelity_report_bytes(name: &str, bytes: &[u8]) -> Result<GraphFidelityReport> {
    let proto = onnx_proto::ModelProto::decode(bytes).map_err(|err| {
        NyError::ModelLoad(format!("graph-fidelity: failed to decode {name}: {err}"))
    })?;
    let graph = proto
        .graph
        .as_ref()
        .ok_or_else(|| NyError::ModelLoad(format!("graph-fidelity: {name} has no graph")))?;

    let loaded = load_onnx_bytes(name, bytes)?;
    let authored = AuthoredWeights::from_graph(graph);
    let folds = attribute_batch_norm_folds(graph, &authored, &loaded.network.layers);

    let tensors = build_tensor_records(graph, &authored, &loaded.weights, &folds);
    let nodes = build_node_records(graph, &authored, &loaded.network.layers, &tensors);

    let authored_batch_norm_nodes = graph
        .node
        .iter()
        .filter(|node| node.op_type == "BatchNormalization")
        .count();
    let post_load_batch_norm_layers = loaded
        .network
        .layers
        .iter()
        .filter(|layer| layer.layer_type == ny_core::LayerType::BatchNorm)
        .count();

    Ok(GraphFidelityReport {
        model: name.to_string(),
        authored_initializers: graph.initializer.len(),
        authored_constants: authored.order.len().saturating_sub(graph.initializer.len()),
        authored_comparable: authored.values.len() + authored.integers.len(),
        authored_undetermined: authored.unreadable.len(),
        post_load_weights: loaded.weights.len(),
        authored_batch_norm_nodes,
        post_load_batch_norm_layers,
        tensors,
        nodes,
    })
}

/// Everything the ONNX authors as a constant — graph initializers and
/// `Constant` node attributes — read straight from the protobuf before any
/// loader rewrite can touch it.
struct AuthoredWeights {
    /// FLOAT payloads, by name.
    values: BTreeMap<String, ArrayD<f32>>,
    /// Integer payloads, by name, flattened in row-major order.
    integers: BTreeMap<String, Vec<i64>>,
    /// Authored but unreadable payloads, with the reason.
    unreadable: BTreeMap<String, String>,
    /// Authored shape per readable name, float and integer alike.
    shapes: BTreeMap<String, Vec<usize>>,
    /// Where each authored name came from.
    kinds: BTreeMap<String, TensorKind>,
    /// Authored names in graph order: initializers, then `Constant` outputs.
    order: Vec<String>,
}

impl AuthoredWeights {
    fn from_graph(graph: &onnx_proto::GraphProto) -> Self {
        let mut out = Self {
            values: BTreeMap::new(),
            integers: BTreeMap::new(),
            unreadable: BTreeMap::new(),
            shapes: BTreeMap::new(),
            kinds: BTreeMap::new(),
            order: Vec::with_capacity(graph.initializer.len()),
        };
        for init in &graph.initializer {
            out.absorb(
                init.name.clone(),
                init,
                TensorKind::FloatInitializer,
                TensorKind::IntegerInitializer,
            );
        }
        // `Constant` nodes: the loader lifts their single output into the same
        // name-keyed store, so their payload is authored too.
        for node in &graph.node {
            if node.op_type != "Constant" || node.output.len() != 1 {
                continue;
            }
            let Some(name) = node.output.first().filter(|name| !name.is_empty()) else {
                continue;
            };
            if let Some(tensor) = node
                .attribute
                .iter()
                .find(|attr| attr.name == "value" && attr.r#type == attribute_type::TENSOR)
                .and_then(|attr| attr.t.as_ref())
            {
                out.absorb(
                    name.clone(),
                    tensor,
                    TensorKind::FloatConstant,
                    TensorKind::IntegerConstant,
                );
                continue;
            }
            if let Some(attr) = node.attribute.iter().find(|attr| {
                (attr.name == "value_float" && attr.r#type == attribute_type::FLOAT)
                    || (attr.name == "value_floats" && attr.r#type == attribute_type::FLOATS)
            }) {
                let values = if attr.name == "value_float" {
                    vec![attr.f_value()]
                } else {
                    attr.floats.clone()
                };
                let shape = if attr.name == "value_float" {
                    Vec::new()
                } else {
                    vec![values.len()]
                };
                if let Ok(tensor) = ArrayD::from_shape_vec(IxDyn(&shape), values) {
                    out.order.push(name.clone());
                    out.kinds.insert(name.clone(), TensorKind::FloatConstant);
                    out.shapes.insert(name.clone(), tensor.shape().to_vec());
                    out.values.insert(name.clone(), tensor);
                }
                continue;
            }
            if let Some(attr) = node.attribute.iter().find(|attr| {
                (attr.name == "value_int" && attr.r#type == attribute_type::INT)
                    || (attr.name == "value_ints" && attr.r#type == attribute_type::INTS)
            }) {
                let values = if attr.name == "value_int" {
                    vec![attr.i_value()]
                } else {
                    attr.ints.clone()
                };
                out.order.push(name.clone());
                out.kinds.insert(name.clone(), TensorKind::IntegerConstant);
                out.shapes.insert(name.clone(), vec![values.len()]);
                out.integers.insert(name.clone(), values);
            }
        }
        out
    }

    fn absorb(
        &mut self,
        name: String,
        tensor: &onnx_proto::TensorProto,
        float_kind: TensorKind,
        integer_kind: TensorKind,
    ) {
        self.order.push(name.clone());
        if tensor.data_location == ONNX_DATA_LOCATION_EXTERNAL {
            self.kinds.insert(name.clone(), TensorKind::Unreadable);
            self.unreadable.insert(
                name,
                "authored payload is external data, not in this file".to_string(),
            );
            return;
        }
        if tensor.data_type == ONNX_TENSOR_FLOAT32 {
            match read_float_initializer(tensor) {
                Some(values) => {
                    self.kinds.insert(name.clone(), float_kind);
                    self.shapes.insert(name.clone(), values.shape().to_vec());
                    self.values.insert(name, values);
                }
                None => {
                    self.kinds.insert(name.clone(), TensorKind::Unreadable);
                    self.unreadable.insert(
                        name,
                        "FLOAT payload does not match its declared dims".to_string(),
                    );
                }
            }
            return;
        }
        match read_integer_initializer(tensor) {
            Some(values) => {
                self.kinds.insert(name.clone(), integer_kind);
                self.shapes.insert(
                    name.clone(),
                    tensor.dims.iter().map(|&dim| dim as usize).collect(),
                );
                self.integers.insert(name, values);
            }
            None => {
                self.kinds.insert(name.clone(), TensorKind::Unreadable);
                self.unreadable.insert(
                    name,
                    format!(
                        "authored dtype {} is neither FLOAT nor an integer payload this tool reads",
                        tensor.data_type
                    ),
                );
            }
        }
    }

    fn vector(&self, name: &str) -> Option<Vec<f64>> {
        Some(
            self.values
                .get(name)?
                .iter()
                .map(|&v| f64::from(v))
                .collect(),
        )
    }

    fn is_authored(&self, name: &str) -> bool {
        self.kinds.contains_key(name)
    }
}

/// Decode an inline FLOAT `TensorProto` without going through the loader.
fn read_float_initializer(init: &onnx_proto::TensorProto) -> Option<ArrayD<f32>> {
    let shape: Vec<usize> = init
        .dims
        .iter()
        .map(|&dim| usize::try_from(dim).ok())
        .collect::<Option<Vec<usize>>>()?;
    let expected: usize = shape.iter().product();
    let values: Vec<f32> = if init.raw_data.is_empty() {
        init.float_data.clone()
    } else {
        if !init.raw_data.len().is_multiple_of(4) {
            return None;
        }
        init.raw_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    };
    if values.len() != expected {
        return None;
    }
    ArrayD::from_shape_vec(IxDyn(&shape), values).ok()
}

/// Decode an inline INT64/INT32 `TensorProto` payload, flattened.
///
/// Integer constants are compared as integers, not against the store's lossy
/// f32 mirror: an `i64 -> f32` widening is not a rewrite.
fn read_integer_initializer(init: &onnx_proto::TensorProto) -> Option<Vec<i64>> {
    const ONNX_TENSOR_INT32: i32 = 6;
    const ONNX_TENSOR_INT64: i32 = 7;
    let expected: usize = init
        .dims
        .iter()
        .map(|&dim| usize::try_from(dim).ok())
        .collect::<Option<Vec<usize>>>()?
        .iter()
        .product();
    let values: Vec<i64> = match init.data_type {
        ONNX_TENSOR_INT64 => {
            if init.raw_data.is_empty() {
                init.int64_data.clone()
            } else {
                if !init.raw_data.len().is_multiple_of(8) {
                    return None;
                }
                init.raw_data
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect()
            }
        }
        ONNX_TENSOR_INT32 => {
            if init.raw_data.is_empty() {
                init.int32_data.iter().map(|&v| i64::from(v)).collect()
            } else {
                if !init.raw_data.len().is_multiple_of(4) {
                    return None;
                }
                init.raw_data
                    .chunks_exact(4)
                    .map(|chunk| {
                        i64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    })
                    .collect()
            }
        }
        _ => return None,
    };
    (values.len() == expected).then_some(values)
}

/// The f64 reference for one fold product.
struct FoldReference {
    batch_norm_node: String,
    host_node: String,
    pattern: &'static str,
    role: &'static str,
    reference: ArrayD<f64>,
}

/// Match every `BatchNormalization` fold pattern the loader implements and build
/// an f64 reference for each fold product.
///
/// Patterns covered, matching `loader/fusion/batch_norm_fold.rs`:
/// `Conv|ConvTranspose|Gemm -> BN`, `Gemm -> Reshape -> BN`, and
/// `BN -> Reshape -> Gemm`.
///
/// This mirrors the loader's fold *equations*, not its guards: the caller only
/// applies a reference to a tensor whose bits actually changed, so a fold the
/// loader declined cannot be misreported. Where two patterns could explain the
/// same tensor the caller attaches neither and says so, rather than guessing.
fn attribute_batch_norm_folds(
    graph: &onnx_proto::GraphProto,
    authored: &AuthoredWeights,
    post_load_layers: &[crate::LayerSpec],
) -> HashMap<String, Vec<FoldReference>> {
    let mut producer: HashMap<&str, usize> = HashMap::new();
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in graph.node.iter().enumerate() {
        for output in &node.output {
            producer.entry(output.as_str()).or_insert(idx);
        }
        for input in &node.input {
            consumers.entry(input.as_str()).or_default().push(idx);
        }
    }
    let sole_consumer = |name: &str| -> Option<&onnx_proto::NodeProto> {
        match consumers.get(name)?.as_slice() {
            [idx] => graph.node.get(*idx),
            _ => None,
        }
    };
    let producer_of =
        |name: &str| -> Option<&onnx_proto::NodeProto> { graph.node.get(*producer.get(name)?) };

    let mut out: HashMap<String, Vec<FoldReference>> = HashMap::new();
    for bn in graph.node.iter() {
        if bn.op_type != "BatchNormalization" || bn.input.len() < 5 {
            continue;
        }
        let Some((scale, shift)) = batch_norm_affine_f64(bn, authored) else {
            continue;
        };
        let Some(bn_input) = bn.input.first().filter(|name| !name.is_empty()) else {
            continue;
        };
        let mut references = Vec::new();

        // Pattern 1: the weight-carrying node is BN's immediate predecessor.
        if let Some(host) = producer_of(bn_input) {
            references.extend(direct_fold_references(
                bn,
                host,
                &scale,
                &shift,
                authored,
                post_load_layers,
            ));

            // Pattern 2: Gemm -> Reshape -> BN. The weights sit behind the
            // Reshape; the channel map is `c(f) = f / block`.
            if host.op_type == "Reshape" {
                if let Some(gemm) = host
                    .input
                    .first()
                    .and_then(|name| producer_of(name))
                    .filter(|node| node.op_type == "Gemm")
                {
                    references.extend(gemm_reshape_bn_references(
                        bn,
                        gemm,
                        &scale,
                        &shift,
                        authored,
                        post_load_layers,
                    ));
                }
            }
        }

        // Pattern 3: BN -> Reshape -> Gemm, folded forward into the Gemm.
        if let Some(reshape) = bn
            .output
            .first()
            .filter(|name| !name.is_empty())
            .and_then(|name| sole_consumer(name))
            .filter(|node| node.op_type == "Reshape")
        {
            if let Some(gemm) = reshape
                .output
                .first()
                .and_then(|name| sole_consumer(name))
                .filter(|node| node.op_type == "Gemm")
            {
                references.extend(bn_reshape_gemm_references(
                    bn,
                    gemm,
                    &scale,
                    &shift,
                    authored,
                    post_load_layers,
                ));
            }
        }

        for (name, reference) in references {
            out.entry(name).or_default().push(reference);
        }
    }
    out
}

/// `Conv|ConvTranspose|Gemm -> BatchNormalization`: a per-output-channel scale
/// of the kernel and `b' = b * scale + shift`.
fn direct_fold_references(
    bn: &onnx_proto::NodeProto,
    host: &onnx_proto::NodeProto,
    scale: &[f64],
    shift: &[f64],
    authored: &AuthoredWeights,
    post_load_layers: &[crate::LayerSpec],
) -> Vec<(String, FoldReference)> {
    let weight_axis = match host.op_type.as_str() {
        "Conv" => 0,
        "ConvTranspose" => 1,
        "Gemm" => usize::from(!gemm_trans_b(host)),
        _ => return Vec::new(),
    };
    let Some(weight_name) = host.input.get(1).filter(|name| !name.is_empty()) else {
        return Vec::new();
    };
    let Some(weight) = authored.values.get(weight_name.as_str()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(reference) = scale_axis_by(weight, weight_axis, scale.len(), |idx| scale[idx]) {
        out.push((
            weight_name.clone(),
            FoldReference {
                batch_norm_node: node_label(bn),
                host_node: node_label(host),
                pattern: "conv/gemm+bn",
                role: "weight",
                reference,
            },
        ));
    }
    if let Some(bias_name) = folded_bias_name(host, weight_name, post_load_layers) {
        let authored_bias = authored.values.get(bias_name.as_str());
        if let Some(reference) =
            affine_bias_reference(authored_bias, scale.len(), |idx| (scale[idx], shift[idx]))
        {
            out.push((
                bias_name,
                FoldReference {
                    batch_norm_node: node_label(bn),
                    host_node: node_label(host),
                    pattern: "conv/gemm+bn",
                    role: "bias",
                    reference,
                },
            ));
        }
    }
    out
}

/// `Gemm -> Reshape -> BatchNormalization`: feature `f` belongs to BN channel
/// `f / block`, so `W'[f, :] = scale[f/block] * W[f, :]` and
/// `b'[f] = scale[f/block] * b[f] + shift[f/block]`.
fn gemm_reshape_bn_references(
    bn: &onnx_proto::NodeProto,
    gemm: &onnx_proto::NodeProto,
    scale: &[f64],
    shift: &[f64],
    authored: &AuthoredWeights,
    post_load_layers: &[crate::LayerSpec],
) -> Vec<(String, FoldReference)> {
    let Some(weight_name) = gemm.input.get(1).filter(|name| !name.is_empty()) else {
        return Vec::new();
    };
    let Some(weight) = authored.values.get(weight_name.as_str()) else {
        return Vec::new();
    };
    if weight.ndim() != 2 {
        return Vec::new();
    }
    let trans_b = gemm_trans_b(gemm);
    let feature_axis = usize::from(!trans_b);
    let Some(block) = channel_block(weight.shape()[feature_axis], scale.len()) else {
        return Vec::new();
    };
    let features = weight.shape()[feature_axis];

    let mut out = Vec::new();
    if let Some(reference) = scale_axis_by(weight, feature_axis, features, |f| scale[f / block]) {
        out.push((
            weight_name.clone(),
            FoldReference {
                batch_norm_node: node_label(bn),
                host_node: node_label(gemm),
                pattern: "gemm->reshape->bn",
                role: "weight",
                reference,
            },
        ));
    }
    if let Some(bias_name) = folded_bias_name(gemm, weight_name, post_load_layers) {
        let authored_bias = authored.values.get(bias_name.as_str());
        if let Some(reference) = affine_bias_reference(authored_bias, features, |f| {
            (scale[f / block], shift[f / block])
        }) {
            out.push((
                bias_name,
                FoldReference {
                    batch_norm_node: node_label(bn),
                    host_node: node_label(gemm),
                    pattern: "gemm->reshape->bn",
                    role: "bias",
                    reference,
                },
            ));
        }
    }
    out
}

/// `BatchNormalization -> Reshape -> Gemm`: `W'[o,f] = W[o,f] * scale[f/block]`
/// and `b'[o] = b[o] + sum_f W[o,f] * shift[f/block]`, the sum taken in
/// ascending `f` exactly as the loader accumulates it.
fn bn_reshape_gemm_references(
    bn: &onnx_proto::NodeProto,
    gemm: &onnx_proto::NodeProto,
    scale: &[f64],
    shift: &[f64],
    authored: &AuthoredWeights,
    post_load_layers: &[crate::LayerSpec],
) -> Vec<(String, FoldReference)> {
    let Some(weight_name) = gemm.input.get(1).filter(|name| !name.is_empty()) else {
        return Vec::new();
    };
    let Some(weight) = authored.values.get(weight_name.as_str()) else {
        return Vec::new();
    };
    if weight.ndim() != 2 {
        return Vec::new();
    }
    let trans_b = gemm_trans_b(gemm);
    let feature_axis = usize::from(trans_b);
    let features = weight.shape()[feature_axis];
    let outputs = weight.shape()[1 - feature_axis];
    let Some(block) = channel_block(features, scale.len()) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    if let Some(reference) = scale_axis_by(weight, feature_axis, features, |f| scale[f / block]) {
        out.push((
            weight_name.clone(),
            FoldReference {
                batch_norm_node: node_label(bn),
                host_node: node_label(gemm),
                pattern: "bn->reshape->gemm",
                role: "weight",
                reference,
            },
        ));
    }
    if let Some(bias_name) = folded_bias_name(gemm, weight_name, post_load_layers) {
        let authored_bias = authored.values.get(bias_name.as_str());
        if authored_bias.is_none_or(|bias| bias.len() == outputs) {
            let mut values = Vec::with_capacity(outputs);
            for output in 0..outputs {
                let mut value = authored_bias
                    .and_then(|bias| bias.iter().nth(output).copied())
                    .map_or(0.0, f64::from);
                for feature in 0..features {
                    let coefficient = if trans_b {
                        weight[[output, feature]]
                    } else {
                        weight[[feature, output]]
                    };
                    value += f64::from(coefficient) * shift[feature / block];
                }
                values.push(value);
            }
            if let Ok(reference) = ArrayD::from_shape_vec(IxDyn(&[outputs]), values) {
                out.push((
                    bias_name,
                    FoldReference {
                        batch_norm_node: node_label(bn),
                        host_node: node_label(gemm),
                        pattern: "bn->reshape->gemm",
                        role: "bias",
                        reference,
                    },
                ));
            }
        }
    }
    out
}

/// `block` such that feature `f` maps to BN channel `f / block`, or `None` when
/// the feature count is not a whole multiple of the channel count.
fn channel_block(features: usize, channels: usize) -> Option<usize> {
    if channels == 0 || features == 0 || !features.is_multiple_of(channels) {
        return None;
    }
    Some(features / channels)
}

/// Name the fold wrote its bias to: the authored C/B input when the host had
/// one, otherwise the fresh name NY synthesized.
///
/// The synthesized name is read off NY's own post-load layer rather than
/// re-derived, and the layer is located by its weight input — which every fold
/// requires to have a single consumer.
fn folded_bias_name(
    host: &onnx_proto::NodeProto,
    weight_name: &str,
    post_load_layers: &[crate::LayerSpec],
) -> Option<String> {
    if let Some(name) = host.input.get(2).filter(|name| !name.is_empty()) {
        return Some(name.clone());
    }
    post_load_layers
        .iter()
        .find(|layer| layer.inputs.get(1).map(String::as_str) == Some(weight_name))
        .and_then(|layer| layer.inputs.get(2))
        .filter(|name| !name.is_empty())
        .cloned()
}

fn gemm_trans_b(node: &onnx_proto::NodeProto) -> bool {
    node.attribute
        .iter()
        .find(|attr| attr.name == "transB")
        .is_some_and(|attr| attr.i_value() != 0)
}

/// Authored node identity: its name, or its first output when unnamed. This is
/// the identity the loader gives the resulting `LayerSpec`.
fn node_label(node: &onnx_proto::NodeProto) -> String {
    if node.name.is_empty() {
        node.output.first().cloned().unwrap_or_default()
    } else {
        node.name.clone()
    }
}

/// The BN affine in f64: `scale = gamma / sqrt(var + eps)`,
/// `shift = beta - gamma * mean / sqrt(var + eps)`.
fn batch_norm_affine_f64(
    node: &onnx_proto::NodeProto,
    authored: &AuthoredWeights,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let gamma = authored.vector(node.input.get(1)?)?;
    let beta = authored.vector(node.input.get(2)?)?;
    let mean = authored.vector(node.input.get(3)?)?;
    let var = authored.vector(node.input.get(4)?)?;
    if gamma.is_empty()
        || gamma.len() != beta.len()
        || gamma.len() != mean.len()
        || gamma.len() != var.len()
    {
        return None;
    }
    let epsilon = f64::from(batch_norm_epsilon(node));
    let mut scale = Vec::with_capacity(gamma.len());
    let mut shift = Vec::with_capacity(gamma.len());
    for idx in 0..gamma.len() {
        let denominator = (var[idx] + epsilon).sqrt();
        if !denominator.is_finite() || denominator <= 0.0 {
            return None;
        }
        scale.push(gamma[idx] / denominator);
        shift.push(beta[idx] - gamma[idx] * mean[idx] / denominator);
    }
    Some((scale, shift))
}

/// The authored epsilon, read the way the loader reads it.
fn batch_norm_epsilon(node: &onnx_proto::NodeProto) -> f32 {
    node.attribute
        .iter()
        .find(|attr| attr.name == "epsilon")
        .and_then(|attr| match attr.r#type {
            attribute_type::FLOAT => Some(attr.f_value()),
            attribute_type::INT => Some(attr.i_value() as f32),
            attribute_type::FLOATS => attr.floats.first().copied(),
            attribute_type::INTS => attr.ints.first().map(|&value| value as f32),
            _ => None,
        })
        .unwrap_or(DEFAULT_BATCH_NORM_EPSILON)
}

/// `W'[.., i, ..] = W[.., i, ..] * factor(i)` in f64, along `axis`.
///
/// `expected_len` pins the axis length the caller's index map assumes, so a
/// mismatched pattern produces no reference instead of a wrong one.
fn scale_axis_by<F: Fn(usize) -> f64>(
    weight: &ArrayD<f32>,
    axis: usize,
    expected_len: usize,
    factor: F,
) -> Option<ArrayD<f64>> {
    if weight.ndim() <= axis || weight.shape()[axis] != expected_len {
        return None;
    }
    let mut reference = weight.mapv(f64::from);
    for (idx, mut lane) in reference.axis_iter_mut(Axis(axis)).enumerate() {
        lane *= factor(idx);
    }
    Some(reference)
}

/// `b'[i] = b[i] * scale(i) + shift(i)` in f64, with an absent bias read as 0.
fn affine_bias_reference<F: Fn(usize) -> (f64, f64)>(
    authored_bias: Option<&ArrayD<f32>>,
    len: usize,
    affine: F,
) -> Option<ArrayD<f64>> {
    // Gemm C may be authored as [1, N] or [N, 1]; the loader normalizes it to
    // [N] before folding. Any other element count is not attributable.
    if authored_bias.is_some_and(|bias| bias.len() != len) {
        return None;
    }
    let mut values = Vec::with_capacity(len);
    for idx in 0..len {
        let (scale, shift) = affine(idx);
        let authored = authored_bias
            .and_then(|bias| bias.iter().nth(idx).copied())
            .map_or(0.0, f64::from);
        values.push(authored * scale + shift);
    }
    ArrayD::from_shape_vec(IxDyn(&[len]), values).ok()
}

/// Which authored nodes consume each tensor, as `op_type:node` labels.
fn tensor_consumers(graph: &onnx_proto::GraphProto) -> HashMap<&str, Vec<String>> {
    let mut out: HashMap<&str, Vec<String>> = HashMap::new();
    for node in &graph.node {
        for input in &node.input {
            if input.is_empty() {
                continue;
            }
            out.entry(input.as_str()).or_default().push(format!(
                "{}:{}",
                node.op_type,
                node_label(node)
            ));
        }
    }
    out
}

fn build_tensor_records(
    graph: &onnx_proto::GraphProto,
    authored: &AuthoredWeights,
    post_load: &WeightStore,
    folds: &HashMap<String, Vec<FoldReference>>,
) -> Vec<TensorFidelity> {
    let consumers = tensor_consumers(graph);
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut records = Vec::new();

    for name in &authored.order {
        if !seen.insert(name.as_str()) {
            continue;
        }
        records.push(tensor_record(
            name, authored, post_load, folds, &consumers, false,
        ));
    }
    // Synthesized weights: in the post-load store, never authored. Sorted for
    // a deterministic report.
    let synthesized: BTreeSet<&str> = post_load
        .keys()
        .filter(|name| !seen.contains(*name))
        .collect();
    for name in synthesized {
        records.push(tensor_record(
            name, authored, post_load, folds, &consumers, true,
        ));
    }
    records
}

fn tensor_record(
    name: &str,
    authored: &AuthoredWeights,
    post_load: &WeightStore,
    folds: &HashMap<String, Vec<FoldReference>>,
    consumers: &HashMap<&str, Vec<String>>,
    synthesized: bool,
) -> TensorFidelity {
    let current = post_load.get(name);
    let authored_value = authored.values.get(name);
    let authored_integers = authored.integers.get(name);
    let kind = if synthesized {
        TensorKind::Synthesized
    } else {
        authored
            .kinds
            .get(name)
            .copied()
            .unwrap_or(TensorKind::Unreadable)
    };
    let mut note = authored.unreadable.get(name).cloned();

    let (status, vs_authored) = if synthesized {
        (TensorStatus::Synthesized, None)
    } else if note.is_some() {
        (TensorStatus::Undetermined, None)
    } else if let Some(want) = authored_integers {
        // Integer payloads are compared exactly, against the store's integer
        // copy. The store also keeps a lossy f32 mirror, which is a widening,
        // not a rewrite.
        match post_load.get_integers(name) {
            Some(have) if have.len() == want.len() => {
                let differing = have
                    .iter()
                    .zip(want)
                    .filter(|(have, want)| have != want)
                    .count();
                let max_abs = have
                    .iter()
                    .zip(want)
                    .map(|(have, want)| (*have as f64 - *want as f64).abs())
                    .fold(0.0, nan_aware_max);
                let deviation = Deviation {
                    elements: want.len(),
                    elements_differing: differing,
                    max_abs,
                    max_rel: 0.0,
                    max_ulp: 0,
                };
                let status = if differing == 0 {
                    TensorStatus::Identical
                } else {
                    TensorStatus::Rewritten
                };
                (status, Some(deviation))
            }
            Some(_) => (TensorStatus::Reshaped, None),
            None => {
                note = Some(
                    "authored as an integer tensor but the post-load store has no integer copy"
                        .to_string(),
                );
                (TensorStatus::Undetermined, None)
            }
        }
    } else {
        match (authored_value, current) {
            (Some(want), Some(have)) if want.shape() == have.shape() => {
                let widened = want.mapv(f64::from);
                let deviation = Deviation::measure(have, &widened);
                let status = match deviation {
                    Some(dev) if dev.is_exact() => TensorStatus::Identical,
                    Some(_) => TensorStatus::Rewritten,
                    None => TensorStatus::Undetermined,
                };
                (status, deviation)
            }
            (Some(_), Some(_)) => (TensorStatus::Reshaped, None),
            (Some(_), None) => (TensorStatus::Dropped, None),
            // An authored name with neither a float nor an integer payload and
            // no recorded reason cannot happen; treat defensively.
            (None, _) => (TensorStatus::Undetermined, None),
        }
    };

    // Attribution explains an observed rewrite only. Two patterns claiming the
    // same tensor is reported, not guessed between.
    let candidates = folds.get(name).map(Vec::as_slice).unwrap_or_default();
    let fold = if status.is_faithful() {
        None
    } else {
        match candidates {
            [reference] => current.and_then(|have| {
                Some(FoldAttribution {
                    batch_norm_node: reference.batch_norm_node.clone(),
                    host_node: reference.host_node.clone(),
                    pattern: reference.pattern.to_string(),
                    role: reference.role.to_string(),
                    deviation: Deviation::measure(have, &reference.reference)?,
                })
            }),
            [] => None,
            many => {
                note = Some(format!(
                    "{} fold patterns could explain this rewrite ({}); attributing none",
                    many.len(),
                    many.iter()
                        .map(|reference| format!("{}:{}", reference.pattern, reference.role))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                None
            }
        }
    };

    TensorFidelity {
        name: name.to_string(),
        kind,
        status,
        authored_shape: authored.shapes.get(name).cloned(),
        post_load_shape: current.map(|tensor| tensor.shape().to_vec()),
        post_load_integer: post_load.get_integers(name).is_some(),
        consumers: consumers.get(name).cloned().unwrap_or_default(),
        vs_authored,
        fold,
        note,
    }
}

/// Severity order for rolling per-tensor statuses up to a node.
fn status_severity(status: TensorStatus) -> u8 {
    match status {
        TensorStatus::Identical => 0,
        TensorStatus::Synthesized => 1,
        TensorStatus::Dropped => 2,
        TensorStatus::Rewritten => 3,
        TensorStatus::Reshaped => 4,
        TensorStatus::Undetermined => 5,
    }
}

fn build_node_records(
    graph: &onnx_proto::GraphProto,
    authored: &AuthoredWeights,
    post_load_layers: &[crate::LayerSpec],
    tensors: &[TensorFidelity],
) -> Vec<NodeFidelity> {
    let by_name: HashMap<&str, &TensorFidelity> = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let layer_names: BTreeSet<&str> = post_load_layers
        .iter()
        .map(|layer| layer.name.as_str())
        .collect();

    graph
        .node
        .iter()
        .map(|node| {
            let label = node_label(node);
            let mut names: Vec<String> = node
                .input
                .iter()
                .filter(|input| authored.is_authored(input))
                .cloned()
                .collect();
            // A Constant node authors its OUTPUT, not an input. Charge the
            // lifted value to it so its row reports whether the value survived.
            if node.op_type == "Constant" {
                names.extend(
                    node.output
                        .iter()
                        .filter(|output| authored.is_authored(output))
                        .cloned(),
                );
            }

            // A node whose identity survives may have gained a synthesized
            // weight input (the fold's new bias). Charge it to this node.
            let surviving = post_load_layers.iter().find(|layer| {
                layer.name == label
                    || layer
                        .inputs
                        .iter()
                        .any(|input| names.iter().any(|name| name == input))
            });
            if let Some(layer) = surviving {
                for input in &layer.inputs {
                    if names.iter().any(|name| name == input) {
                        continue;
                    }
                    if by_name
                        .get(input.as_str())
                        .is_some_and(|tensor| tensor.status == TensorStatus::Synthesized)
                    {
                        names.push(input.clone());
                    }
                }
            }

            let mut status = TensorStatus::Identical;
            let mut vs_authored = Deviation::default();
            let mut vs_fold: Option<Deviation> = None;
            for name in &names {
                let Some(tensor) = by_name.get(name.as_str()) else {
                    continue;
                };
                if status_severity(tensor.status) > status_severity(status) {
                    status = tensor.status;
                }
                if let Some(dev) = tensor.vs_authored {
                    vs_authored = vs_authored.worst(dev);
                }
                if let Some(fold) = tensor.fold.as_ref() {
                    vs_fold = Some(match vs_fold {
                        Some(current) => current.worst(fold.deviation),
                        None => fold.deviation,
                    });
                }
            }

            NodeFidelity {
                node: label.clone(),
                op_type: node.op_type.clone(),
                present_post_load: layer_names.contains(label.as_str()) || surviving.is_some(),
                tensors: names,
                status,
                vs_authored,
                vs_fold_reference: vs_fold,
            }
        })
        .collect()
}

/// Render the report as a fixed-width text summary.
///
/// `all` includes every node and tensor; otherwise only the non-faithful ones.
#[must_use]
pub fn format_report(report: &GraphFidelityReport, all: bool) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let verdict = if report.is_authored_graph() {
        "AUTHORED GRAPH — post-load weights are bit-identical to the ONNX payloads".to_string()
    } else if report.float_weights_are_authored() {
        format!(
            "AUTHORED FLOAT WEIGHTS — bit-identical; {} structural integer tensor(s) added by constant folding",
            report.structural_additions()
        )
    } else {
        "NOT THE AUTHORED GRAPH — NY certifies a load-time rewrite of this network".to_string()
    };
    let _ = writeln!(out, "graph-fidelity gate: {}", report.model);
    let _ = writeln!(out, "  VERDICT: {verdict}");
    let _ = writeln!(
        out,
        "  authored payloads       : {} initializers + {} Constant nodes (comparable {}, undetermined {})",
        report.authored_initializers,
        report.authored_constants,
        report.authored_comparable,
        report.authored_undetermined
    );
    let _ = writeln!(
        out,
        "  post-load weights       : {}",
        report.post_load_weights
    );
    let _ = writeln!(
        out,
        "  identical {} | rewritten {} | reshaped {} | dropped {} | synthesized {} | undetermined {}",
        report.count(TensorStatus::Identical),
        report.count(TensorStatus::Rewritten),
        report.count(TensorStatus::Reshaped),
        report.count(TensorStatus::Dropped),
        report.count(TensorStatus::Synthesized),
        report.count(TensorStatus::Undetermined),
    );
    let _ = writeln!(
        out,
        "  BatchNormalization nodes: {} authored -> {} post-load layers",
        report.authored_batch_norm_nodes, report.post_load_batch_norm_layers
    );
    let _ = writeln!(
        out,
        "  authored nodes removed  : {} of {} (weights they consumed may still sit in the store, orphaned)",
        report.removed_nodes(),
        report.nodes.len(),
    );
    let authored_worst = report.worst_vs_authored();
    let _ = writeln!(
        out,
        "  worst rewrite vs authored ONNX : max_abs={:.6e} max_rel={:.6e} max_ulp={} ({} elements differ)",
        authored_worst.max_abs,
        authored_worst.max_rel,
        authored_worst.max_ulp,
        authored_worst.elements_differing,
    );
    let fold_worst = report.worst_vs_fold_reference();
    let _ = writeln!(
        out,
        "  worst rewrite vs f64 fold ref  : max_abs={:.6e} max_rel={:.6e} max_ulp={} ({} elements differ)",
        fold_worst.max_abs,
        fold_worst.max_rel,
        fold_worst.max_ulp,
        fold_worst.elements_differing,
    );
    for role in ["weight", "bias"] {
        let worst = report.worst_fold_deviation_for_role(role);
        if worst.elements == 0 {
            continue;
        }
        let _ = writeln!(
            out,
            "    fold {:<6} only              : max_abs={:.6e} max_rel={:.6e} max_ulp={}",
            role, worst.max_abs, worst.max_rel, worst.max_ulp,
        );
    }
    let unexplained = report.unexplained_rewrites();
    let _ = writeln!(
        out,
        "  unexplained rewrites    : {}{}",
        unexplained.len(),
        if unexplained.is_empty() {
            String::new()
        } else {
            format!(
                " [{}]",
                unexplained
                    .iter()
                    .map(|tensor| tensor.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    );

    let _ = writeln!(
        out,
        "\n  {:<28} {:<16} {:<13} {:>12} {:>12} {:>8} {:>12} {:>12} {:>8}",
        "node",
        "op",
        "status",
        "auth_abs",
        "auth_rel",
        "auth_ulp",
        "fold_abs",
        "fold_rel",
        "fold_ulp"
    );
    for node in &report.nodes {
        if !all && node.is_faithful() {
            continue;
        }
        let (fold_abs, fold_rel, fold_ulp) = match node.vs_fold_reference {
            Some(dev) => (
                format!("{:.4e}", dev.max_abs),
                format!("{:.4e}", dev.max_rel),
                dev.max_ulp.to_string(),
            ),
            None => ("-".to_string(), "-".to_string(), "-".to_string()),
        };
        let _ = writeln!(
            out,
            "  {:<28} {:<16} {:<13} {:>12.4e} {:>12.4e} {:>8} {:>12} {:>12} {:>8}",
            truncate(&node.node, 28),
            truncate(&node.op_type, 16),
            node.label(),
            node.vs_authored.max_abs,
            node.vs_authored.max_rel,
            node.vs_authored.max_ulp,
            fold_abs,
            fold_rel,
            fold_ulp,
        );
    }

    if all {
        let _ = writeln!(out, "\n  tensors:");
        for tensor in &report.tensors {
            let fold = tensor
                .fold
                .as_ref()
                .map(|fold| {
                    format!(
                        " fold[{} {} of {} <- {}] abs={:.4e} rel={:.4e} ulp={}",
                        fold.pattern,
                        fold.role,
                        fold.host_node,
                        fold.batch_norm_node,
                        fold.deviation.max_abs,
                        fold.deviation.max_rel,
                        fold.deviation.max_ulp,
                    )
                })
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "  {:<40} {:<11} {:<13} authored={:?} post_load={:?}{}{}",
                truncate(&tensor.name, 40),
                tensor.kind.label(),
                tensor.status.label(),
                tensor.authored_shape,
                tensor.post_load_shape,
                fold,
                tensor
                    .note
                    .as_ref()
                    .map(|note| format!(" ({note})"))
                    .unwrap_or_default(),
            );
        }
    }
    out
}

fn truncate(value: &str, width: usize) -> String {
    if value.chars().count() <= width {
        return value.to_string();
    }
    let keep = width.saturating_sub(3);
    format!("{}...", value.chars().take(keep).collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
        let dim = shape
            .iter()
            .map(|&size| onnx_proto::tensor_shape_proto::Dimension {
                value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                    size,
                )),
            })
            .collect();
        onnx_proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx_proto::TypeProto {
                tensor_type: Some(onnx_proto::TensorTypeProto {
                    elem_type: ONNX_TENSOR_FLOAT32,
                    shape: Some(onnx_proto::TensorShapeProto { dim }),
                }),
            }),
        }
    }

    fn initializer(name: &str, dims: &[i64], values: &[f32]) -> onnx_proto::TensorProto {
        onnx_proto::TensorProto {
            dims: dims.to_vec(),
            data_type: ONNX_TENSOR_FLOAT32,
            segment: None,
            name: name.to_string(),
            raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            float_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
            double_data: Vec::new(),
            string_data: Vec::new(),
            uint64_data: Vec::new(),
            external_data: Vec::new(),
            data_location: 0,
        }
    }

    fn int64_initializer(name: &str, dims: &[i64], values: &[i64]) -> onnx_proto::TensorProto {
        onnx_proto::TensorProto {
            dims: dims.to_vec(),
            data_type: 7, // INT64
            segment: None,
            name: name.to_string(),
            raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            float_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
            double_data: Vec::new(),
            string_data: Vec::new(),
            uint64_data: Vec::new(),
            external_data: Vec::new(),
            data_location: 0,
        }
    }

    fn tensor_attr(name: &str, tensor: onnx_proto::TensorProto) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            t: Some(tensor),
            r#type: attribute_type::TENSOR,
            ..Default::default()
        }
    }

    fn node(
        name: &str,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attribute: Vec<onnx_proto::AttributeProto>,
    ) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: inputs.iter().map(|s| (*s).to_string()).collect(),
            output: outputs.iter().map(|s| (*s).to_string()).collect(),
            name: name.to_string(),
            op_type: op_type.to_string(),
            domain: String::new(),
            attribute,
        }
    }

    fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            i: Some(value),
            r#type: attribute_type::INT,
            ..Default::default()
        }
    }

    fn ints_attr(name: &str, values: &[i64]) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            r#type: attribute_type::INTS,
            ints: values.to_vec(),
            ..Default::default()
        }
    }

    fn encode(graph: onnx_proto::GraphProto) -> Vec<u8> {
        onnx_proto::ModelProto {
            ir_version: 9,
            opset_import: vec![onnx_proto::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "ny-fidelity-test".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(graph),
        }
        .encode_to_vec()
    }

    /// Conv(1->1, 2x2 kernel, no bias) -> BatchNormalization -> Relu.
    /// Values are chosen so `gamma / sqrt(var + eps)` is not a power of two and
    /// the f32 product is not exact.
    // #bn-fold-restore: the fold-attribution tests below are the pre-quarantine
    // versions (d1282d23). The quarantine-era tests asserting BatchNorm stays
    // raw under the DEFAULT policy were removed together with the hard
    // `BATCH_NORM_AFFINE_COMPOSITION_AUTHENTICATED = false` gate they pinned;
    // the raw path's properties are still tested, under an explicit
    // `BatchNormFoldingPolicy::PreserveRaw` load, in
    // loader/fusion/tests/batch_norm_ort_prop.rs.

    #[ntest::timeout(60000)]
    #[test]
    fn conv_bn_fold_is_reported_as_a_rewrite() {
        let bytes = encode(conv_bn_graph());
        let report = graph_fidelity_report_bytes("conv_bn.onnx", &bytes).expect("fidelity report");

        assert!(
            !report.is_authored_graph(),
            "the BN fold rewrites conv.weight: {report:#?}"
        );
        assert!(
            !report.float_weights_are_authored(),
            "a folded Conv kernel is a FLOAT rewrite, not a structural one: {report:#?}"
        );
        assert_eq!(report.authored_batch_norm_nodes, 1);
        assert_eq!(report.post_load_batch_norm_layers, 0);

        let weight = report
            .tensors
            .iter()
            .find(|tensor| tensor.name == "conv.weight")
            .expect("conv.weight record");
        assert_eq!(weight.status, TensorStatus::Rewritten);
        let vs_authored = weight.vs_authored.expect("authored deviation");
        assert_eq!(vs_authored.elements, 4);
        assert_eq!(vs_authored.elements_differing, 4);

        // The BN node is gone, but the fold does not evict its four statistics
        // from the weight store: they stay bit-identical and orphaned. The node
        // roll-up is what has to surface the removal.
        for name in ["bn.gamma", "bn.beta", "bn.mean", "bn.var"] {
            let record = report
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap_or_else(|| panic!("{name} record"));
            assert_eq!(record.status, TensorStatus::Identical, "{name}");
        }
        let bn_node = report
            .nodes
            .iter()
            .find(|node| node.node == "BN_1")
            .expect("BN_1 node record");
        assert!(
            !bn_node.present_post_load,
            "the folded BN node must be reported as removed: {bn_node:#?}"
        );
        assert!(!bn_node.is_faithful(), "a removed node is not faithful");
        assert_eq!(bn_node.label(), "node-removed");
        assert_eq!(report.removed_nodes(), 1);

        // The fold synthesizes a bias the authored graph never had.
        assert_eq!(report.count(TensorStatus::Synthesized), 1);
        assert!(
            report.unexplained_rewrites().is_empty(),
            "every rewrite must be attributed to the BN fold: {:?}",
            report
                .unexplained_rewrites()
                .iter()
                .map(|tensor| &tensor.name)
                .collect::<Vec<_>>()
        );
    }

    #[ntest::timeout(60000)]
    #[test]
    fn gemm_bn_fold_respects_trans_b() {
        // transB=1: weight is (out, in) and the BN scale applies to axis 0. A
        // reference on the wrong axis would not match NY's f32 output to
        // within ULPs, so this pins the axis choice.
        let graph = onnx_proto::GraphProto {
            node: vec![
                node(
                    "Gemm_0",
                    "Gemm",
                    &["input", "fc.weight"],
                    &["gemm_out"],
                    vec![int_attr("transB", 1)],
                ),
                node(
                    "BN_1",
                    "BatchNormalization",
                    &["gemm_out", "bn.gamma", "bn.beta", "bn.mean", "bn.var"],
                    &["output"],
                    Vec::new(),
                ),
            ],
            name: "gemm_bn".to_string(),
            initializer: vec![
                initializer("fc.weight", &[2, 3], &[0.1, -0.3, 0.7, 1.1, 0.37, -0.93]),
                initializer("bn.gamma", &[2], &[1.3, 0.61]),
                initializer("bn.beta", &[2], &[-0.7, 0.21]),
                initializer("bn.mean", &[2], &[0.31, -0.12]),
                initializer("bn.var", &[2], &[0.77, 1.31]),
            ],
            sparse_initializer: Vec::new(),
            input: vec![value_info("input", &[1, 3])],
            output: vec![value_info("output", &[1, 2])],
            #[cfg(feature = "onnx-value-info")]
            value_info: Vec::new(),
        };
        let bytes = encode(graph);
        let report = graph_fidelity_report_bytes("gemm_bn.onnx", &bytes).expect("fidelity report");

        let weight = report
            .tensors
            .iter()
            .find(|tensor| tensor.name == "fc.weight")
            .expect("fc.weight record");
        assert_eq!(weight.status, TensorStatus::Rewritten);
        let fold = weight.fold.as_ref().expect("fold attribution");
        assert_eq!(fold.pattern, "conv/gemm+bn");
        assert!(
            fold.deviation.max_ulp <= 2,
            "transB=1 scale axis must be axis 0: {:?}",
            fold.deviation
        );
    }

    #[ntest::timeout(60000)]
    #[test]
    fn gemm_reshape_bn_fold_is_attributed_across_the_reshape() {
        // Gemm[1,6] -> Reshape[-1,2,3] -> BN(C=2): feature f lands in channel
        // f / 3, so the reference must apply scale[f/3], not scale[f]. A wrong
        // block map is an O(1) deviation, not a ULP-level one.
        let graph = onnx_proto::GraphProto {
            node: vec![
                node(
                    "Gemm_0",
                    "Gemm",
                    &["input", "fc.weight"],
                    &["gemm_out"],
                    vec![int_attr("transB", 1)],
                ),
                node(
                    "Reshape_1",
                    "Reshape",
                    &["gemm_out", "reshape.shape"],
                    &["reshaped"],
                    Vec::new(),
                ),
                node(
                    "BN_2",
                    "BatchNormalization",
                    &["reshaped", "bn.gamma", "bn.beta", "bn.mean", "bn.var"],
                    &["output"],
                    Vec::new(),
                ),
            ],
            name: "gemm_reshape_bn".to_string(),
            initializer: vec![
                initializer(
                    "fc.weight",
                    &[6, 2],
                    &[
                        0.1, -0.3, 0.7, 1.1, 0.37, -0.93, 0.512, -0.061, 1.7, 0.23, -0.44, 0.89,
                    ],
                ),
                int64_initializer("reshape.shape", &[3], &[-1, 2, 3]),
                initializer("bn.gamma", &[2], &[1.3, 0.61]),
                initializer("bn.beta", &[2], &[-0.7, 0.21]),
                initializer("bn.mean", &[2], &[0.31, -0.12]),
                initializer("bn.var", &[2], &[0.77, 1.31]),
            ],
            sparse_initializer: Vec::new(),
            input: vec![value_info("input", &[1, 2])],
            output: vec![value_info("output", &[1, 2, 3])],
            #[cfg(feature = "onnx-value-info")]
            value_info: Vec::new(),
        };
        let bytes = encode(graph);
        // The extended folds are default-ON but env-gated; other tests in this
        // binary walk that gate, so take the same lock they do.
        let report =
            ny_test_utils::env::with_serialized_env_vars_removed(&["NY_BN_FOLD_EXT"], || {
                graph_fidelity_report_bytes("gemm_reshape_bn.onnx", &bytes)
            })
            .expect("fidelity report");

        let weight = report
            .tensors
            .iter()
            .find(|tensor| tensor.name == "fc.weight")
            .expect("fc.weight record");
        assert_eq!(weight.status, TensorStatus::Rewritten);
        let fold = weight.fold.as_ref().expect("fold attribution");
        assert_eq!(fold.pattern, "gemm->reshape->bn");
        assert_eq!(fold.host_node, "Gemm_0");
        assert_eq!(fold.batch_norm_node, "BN_2");
        assert!(
            fold.deviation.max_rel < 1e-6 && fold.deviation.max_ulp <= 4,
            "the f = c * block channel map must match the loader's: {:?}",
            fold.deviation
        );
        assert!(
            report.unexplained_rewrites().is_empty(),
            "the across-Reshape fold must be attributed: {:?}",
            report
                .unexplained_rewrites()
                .iter()
                .map(|tensor| &tensor.name)
                .collect::<Vec<_>>()
        );
    }

    fn conv_bn_graph() -> onnx_proto::GraphProto {
        onnx_proto::GraphProto {
            node: vec![
                node(
                    "Conv_0",
                    "Conv",
                    &["input", "conv.weight"],
                    &["conv_out"],
                    vec![ints_attr("kernel_shape", &[2, 2])],
                ),
                node(
                    "BN_1",
                    "BatchNormalization",
                    &["conv_out", "bn.gamma", "bn.beta", "bn.mean", "bn.var"],
                    &["bn_out"],
                    Vec::new(),
                ),
                node("Relu_2", "Relu", &["bn_out"], &["output"], Vec::new()),
            ],
            name: "conv_bn".to_string(),
            initializer: vec![
                initializer("conv.weight", &[1, 1, 2, 2], &[0.1, -0.3, 0.7, 1.234_567_9]),
                initializer("bn.gamma", &[1], &[1.3]),
                initializer("bn.beta", &[1], &[-0.7]),
                initializer("bn.mean", &[1], &[0.31]),
                initializer("bn.var", &[1], &[0.77]),
            ],
            sparse_initializer: Vec::new(),
            input: vec![value_info("input", &[1, 1, 3, 3])],
            output: vec![value_info("output", &[1, 1, 2, 2])],
            #[cfg(feature = "onnx-value-info")]
            value_info: Vec::new(),
        }
    }

    /// The same Conv, with no BatchNormalization to fold.
    fn conv_only_graph() -> onnx_proto::GraphProto {
        onnx_proto::GraphProto {
            node: vec![
                node(
                    "Conv_0",
                    "Conv",
                    &["input", "conv.weight", "conv.bias"],
                    &["conv_out"],
                    vec![ints_attr("kernel_shape", &[2, 2])],
                ),
                node("Relu_1", "Relu", &["conv_out"], &["output"], Vec::new()),
            ],
            name: "conv_only".to_string(),
            initializer: vec![
                initializer("conv.weight", &[1, 1, 2, 2], &[0.1, -0.3, 0.7, 1.234_567_9]),
                initializer("conv.bias", &[1], &[0.25]),
            ],
            sparse_initializer: Vec::new(),
            input: vec![value_info("input", &[1, 1, 3, 3])],
            output: vec![value_info("output", &[1, 1, 2, 2])],
            #[cfg(feature = "onnx-value-info")]
            value_info: Vec::new(),
        }
    }

    #[ntest::timeout(60000)]
    #[test]
    fn conv_only_model_is_the_authored_graph() {
        let bytes = encode(conv_only_graph());
        let report =
            graph_fidelity_report_bytes("conv_only.onnx", &bytes).expect("fidelity report");

        assert!(
            report.is_authored_graph(),
            "no rewrite is expected: {report:#?}"
        );
        assert!(report.float_weights_are_authored());
        assert_eq!(report.count(TensorStatus::Identical), 2);
        assert_eq!(report.count(TensorStatus::Rewritten), 0);
        assert_eq!(report.count(TensorStatus::Synthesized), 0);
        assert_eq!(report.count(TensorStatus::Dropped), 0);
        assert_eq!(report.worst_vs_authored(), Deviation::default());
        assert!(report.unexplained_rewrites().is_empty());
    }

    #[ntest::timeout(60000)]
    #[test]
    fn constant_folded_shape_tensors_do_not_condemn_the_float_weights() {
        // Shape -> Gather -> Unsqueeze -> Concat -> Reshape is folded to integer
        // store entries at load. Measured on dist_shift_2023/mnist_prior.onnx,
        // where all 8 f32 weights stay bit-identical and 4 integer tensors
        // appear. The float-weight claim must survive that; the strict
        // everything-identical claim must not.
        let graph = onnx_proto::GraphProto {
            node: vec![
                node("Shape_0", "Shape", &["input"], &["shape_out"], Vec::new()),
                node(
                    "Constant_1",
                    "Constant",
                    &[],
                    &["gather_idx"],
                    vec![tensor_attr(
                        "value",
                        int64_initializer("idx_value", &[], &[0]),
                    )],
                ),
                node(
                    "Gather_2",
                    "Gather",
                    &["shape_out", "gather_idx"],
                    &["batch"],
                    Vec::new(),
                ),
                node(
                    "Unsqueeze_3",
                    "Unsqueeze",
                    &["batch", "unsqueeze_axes"],
                    &["batch_1d"],
                    Vec::new(),
                ),
                node(
                    "Concat_4",
                    "Concat",
                    &["batch_1d", "tail_dims"],
                    &["target_shape"],
                    vec![int_attr("axis", 0)],
                ),
                node(
                    "Reshape_5",
                    "Reshape",
                    &["input", "target_shape"],
                    &["flat"],
                    Vec::new(),
                ),
                node(
                    "Gemm_6",
                    "Gemm",
                    &["flat", "fc.weight", "fc.bias"],
                    &["output"],
                    vec![int_attr("transB", 1)],
                ),
            ],
            name: "const_folded_shape".to_string(),
            initializer: vec![
                int64_initializer("unsqueeze_axes", &[1], &[0]),
                int64_initializer("tail_dims", &[1], &[4]),
                initializer(
                    "fc.weight",
                    &[2, 4],
                    &[0.1, -0.3, 0.7, 1.1, 0.37, -0.93, 0.5, 0.25],
                ),
                initializer("fc.bias", &[2], &[0.25, -0.125]),
            ],
            sparse_initializer: Vec::new(),
            input: vec![value_info("input", &[1, 4])],
            output: vec![value_info("output", &[1, 2])],
            #[cfg(feature = "onnx-value-info")]
            value_info: Vec::new(),
        };
        let bytes = encode(graph);
        let report = graph_fidelity_report_bytes("const_folded_shape.onnx", &bytes)
            .expect("fidelity report");

        for name in ["fc.weight", "fc.bias"] {
            let record = report
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap_or_else(|| panic!("{name} record"));
            assert_eq!(record.status, TensorStatus::Identical, "{name}");
            assert!(record.kind.is_float_weight(), "{name}");
        }
        assert!(
            report.float_weights_are_authored(),
            "float coefficients are untouched: {report:#?}"
        );
        // Every synthesized tensor here is an integer shape tensor.
        assert_eq!(
            report.structural_additions(),
            report.count(TensorStatus::Synthesized),
            "constant folding must only add integer tensors here: {report:#?}"
        );
    }

    #[ntest::timeout(60000)]
    #[test]
    fn lifted_integer_constant_is_not_a_rewrite() {
        // The loader turns a Constant node into a weight-store entry. Reading
        // initializers only would call that a synthesized weight and condemn an
        // otherwise faithful model (measured on soundnessbench/model.onnx).
        let graph = onnx_proto::GraphProto {
            node: vec![
                node(
                    "Conv_0",
                    "Conv",
                    &["input", "conv.weight", "conv.bias"],
                    &["conv_out"],
                    vec![ints_attr("kernel_shape", &[2, 2])],
                ),
                node(
                    "Shape_1",
                    "Constant",
                    &[],
                    &["shape_c"],
                    vec![tensor_attr(
                        "value",
                        int64_initializer("shape_value", &[2], &[1, 4]),
                    )],
                ),
                node(
                    "Reshape_2",
                    "Reshape",
                    &["conv_out", "shape_c"],
                    &["output"],
                    Vec::new(),
                ),
            ],
            name: "conv_reshape_const".to_string(),
            initializer: vec![
                initializer("conv.weight", &[1, 1, 2, 2], &[0.1, -0.3, 0.7, 1.234_567_9]),
                initializer("conv.bias", &[1], &[0.25]),
            ],
            sparse_initializer: Vec::new(),
            input: vec![value_info("input", &[1, 1, 3, 3])],
            output: vec![value_info("output", &[1, 4])],
            #[cfg(feature = "onnx-value-info")]
            value_info: Vec::new(),
        };
        let bytes = encode(graph);
        let report = graph_fidelity_report_bytes("conv_reshape_const.onnx", &bytes)
            .expect("fidelity report");

        let lifted = report
            .tensors
            .iter()
            .find(|tensor| tensor.name == "shape_c")
            .expect("shape_c record");
        assert_eq!(lifted.kind, TensorKind::IntegerConstant);
        assert_eq!(lifted.status, TensorStatus::Identical);
        assert_eq!(report.count(TensorStatus::Synthesized), 0);
        assert_eq!(
            report.removed_nodes(),
            0,
            "a lifted Constant is not removed"
        );
        assert!(
            report.is_authored_graph(),
            "lifting a Constant is a representation change, not a rewrite: {report:#?}"
        );
    }

    #[test]
    fn ulp_distance_is_signed_zero_blind() {
        assert_eq!(ulp_distance(0.0, -0.0), 0);
        assert_eq!(ulp_distance(1.0, 1.0), 0);
        assert_eq!(ulp_distance(1.0, f32::from_bits(1.0f32.to_bits() + 1)), 1);
        assert_eq!(ulp_distance(f32::NAN, 1.0), u32::MAX);
    }

    #[test]
    fn deviation_measure_rejects_shape_mismatch() {
        let current = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0f32, 2.0]).expect("current");
        let reference = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0f64, 2.0, 3.0]).expect("ref");
        assert!(Deviation::measure(&current, &reference).is_none());
    }
}
