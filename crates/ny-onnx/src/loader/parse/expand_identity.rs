// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cctsdb-expand-identity: certification of PyTorch's
//! `Tensor.expand(-1, -1, ...)` keep-dims idiom over a RUNTIME shape.
//!
//! The exporter's keep-dims `expand` lowers to
//!
//!     s    = Shape(x)                             # runtime dims, each >= 0
//!     ones = ConstantOfShape(Shape(s), value=1)   # same length as s
//!     neg  = Mul(ones, Constant(-1))              # elementwise -1
//!     cond = Equal(s, neg)                        # CONSTANT all-false
//!     out  = Where(cond, ones, s)                 # pass-through of s
//!
//! `cond` is decided by ONNX semantics alone: a `Shape` output is a vector of
//! tensor dimensions, every one of which is `>= 0`, and the other operand is
//! provably elementwise NEGATIVE, so no element can compare equal — for every
//! runtime input, whatever the dims turn out to be.  The paired `Where` then
//! selects its `Y` operand in every lane, and because `cond`, `X` and `Y` all
//! carry the exact shape of `s` by construction, the node is a value-identity
//! of `Y`.  Neither fact depends on the runtime shape values, which is what
//! distinguishes this admission from a blanket dtype relaxation.
//!
//! This mirrors the `is_exact_integer_literal_node` precedent (709f23b69,
//! ml4acopf's literal `Tensor.expand` cone) for the case that cone machinery
//! deliberately cannot cover: here the `Shape` operand is RUNTIME-rooted
//! (`literal_cone` excludes `Shape` on purpose, and the cone touches the graph
//! input), so `require_all_folded`'s WeightStore obligation can never be met.
//! Instead the schema gates DEFER their refusal via
//! [`super::literal_cone::LiteralExemptions::record_canonicalizable`], and
//! [`canonicalize_expand_identity_cones`] discharges it during `prepare` by
//! rewiring every consumer of the certified `Where`'s output to read the
//! pass-through tensor `Y` directly, deleting the then consumer-free `Where`,
//! `Equal`, and their private cones.  Rewiring (not an `Identity` shim) is the
//! continuity idiom `canonicalize_reversed_slice` uses: the surviving graph
//! reads only names that still have producers.  A runtime `Identity` would be
//! dropped by `op_map` with no name-forwarding, leaving the erased `Where`
//! output dangling at graph build (patch-1's `Expand_94` reading `Where_93`'s
//! '328'), and it would also hide the `Shape` output from the direct
//! `Expand(source, Shape(reference))` authentication in `convert.rs`.  A
//! pattern that fails to canonicalize re-raises the original refusal verbatim
//! — fail closed.
//!
//! Deliberately NOT certified (meaningful negatives, pinned by tests):
//! `Equal(Shape(x), 0)` (a dim CAN be 0), `Equal(runtime, runtime)`, a
//! `ConstantOfShape` fill that is zero/negative/float, and a `Where` whose `Y`
//! is not the very `Shape` output the `Equal` compares.

use crate::loader::const_fold::is_standard_onnx_domain;
use crate::onnx_proto::{GraphProto, NodeProto};
use std::collections::{HashMap, HashSet};
use tracing::debug;

use super::quantization_preflight::{exact_integer_constant, exact_integer_initializer};

const INT32: i32 = 6;
const INT64: i32 = 7;

/// A `Where` node proven to be a value-identity of its `Y` input.
struct CertifiedWhere {
    node_index: usize,
    pass_through: String,
}

/// Identity of a node eligible for dead-cone deletion after canonicalization:
/// `(op_type, node name, first output)`.  Certified cone outputs each have
/// exactly one producer, so the triple is unambiguous.
type NodeKey = (String, String, String);

/// Every certified expand-identity `Equal`/`Where` in one authored graph.
pub(super) struct ExpandIdentityCones {
    /// Certified all-false `Equal` outputs, keyed to the `Shape`-node output
    /// they compare against a provably negative operand.
    equal_shape_operand: HashMap<String, String>,
    /// Outputs of `Where` nodes certified as pass-throughs of `Y`.
    where_outputs: HashSet<String>,
    where_rewrites: Vec<CertifiedWhere>,
    /// Cone nodes (Equal, Mul, ConstantOfShape, inner Shape, Constant) that
    /// may be deleted once canonicalization leaves them consumer-free.
    delete_candidates: HashSet<NodeKey>,
}

impl ExpandIdentityCones {
    pub(super) fn certify(graph: &GraphProto) -> Self {
        let mut producer_count: HashMap<&str, usize> = HashMap::new();
        let mut producer_index: HashMap<&str, usize> = HashMap::new();
        for (index, node) in graph.node.iter().enumerate() {
            for output in node.output.iter().filter(|output| !output.is_empty()) {
                *producer_count.entry(output.as_str()).or_default() += 1;
                producer_index.insert(output.as_str(), index);
            }
        }
        let mut initializer_count: HashMap<&str, usize> = HashMap::new();
        for initializer in &graph.initializer {
            if !initializer.name.is_empty() {
                *initializer_count
                    .entry(initializer.name.as_str())
                    .or_default() += 1;
            }
        }
        let context = CertifyContext {
            graph,
            producer_count,
            producer_index,
            initializer_count,
        };

        let mut cones = Self {
            equal_shape_operand: HashMap::new(),
            where_outputs: HashSet::new(),
            where_rewrites: Vec::new(),
            delete_candidates: HashSet::new(),
        };
        for node in &graph.node {
            if let Some(certified) = certify_equal(&context, node) {
                cones
                    .equal_shape_operand
                    .insert(node.output[0].clone(), certified.shape_operand);
                cones.delete_candidates.insert(node_key(node));
                for cone_index in certified.negative_cone {
                    cones
                        .delete_candidates
                        .insert(node_key(&graph.node[cone_index]));
                }
            }
        }
        for (index, node) in graph.node.iter().enumerate() {
            if let Some(certified) = certify_where(&context, &cones.equal_shape_operand, node) {
                cones.where_outputs.insert(node.output[0].clone());
                cones.where_rewrites.push(CertifiedWhere {
                    node_index: index,
                    pass_through: certified.pass_through,
                });
                for cone_index in certified.select_cone {
                    cones
                        .delete_candidates
                        .insert(node_key(&graph.node[cone_index]));
                }
            }
        }
        cones
    }

    /// Whether this node is a certified all-false expand-identity `Equal`.
    pub(super) fn is_certified_equal(&self, node: &NodeProto) -> bool {
        node.op_type == "Equal"
            && node
                .output
                .first()
                .is_some_and(|output| self.equal_shape_operand.contains_key(output))
    }

    /// Whether this node is a certified pass-through expand-identity `Where`.
    pub(super) fn is_certified_where(&self, node: &NodeProto) -> bool {
        node.op_type == "Where"
            && node
                .output
                .first()
                .is_some_and(|output| self.where_outputs.contains(output))
    }

    /// Whether this node belongs to a certified cone the canonicalization
    /// deletes (the `Equal`, its `Mul`/`ConstantOfShape`/inner-`Shape`/
    /// `Constant` operands, and the `Where`'s select cone).  Used by the
    /// arithmetic FLOAT32 gate: such a node (cctsdb's `Mul_91 = ones * -1`)
    /// computes a value whose only consumer is the certified comparison, so
    /// its refusal defers to the same canonicalization ratchet.
    pub(super) fn is_certified_cone_node(&self, node: &NodeProto) -> bool {
        self.delete_candidates.contains(&node_key(node))
    }
}

struct CertifyContext<'graph> {
    graph: &'graph GraphProto,
    producer_count: HashMap<&'graph str, usize>,
    producer_index: HashMap<&'graph str, usize>,
    initializer_count: HashMap<&'graph str, usize>,
}

impl<'graph> CertifyContext<'graph> {
    /// The unique producing node of `value`, when `value` is a plain SSA node
    /// output (exactly one producer, not also an initializer).
    fn single_producer(&self, value: &str) -> Option<&'graph NodeProto> {
        if self.producer_count.get(value).copied() != Some(1)
            || self.initializer_count.contains_key(value)
        {
            return None;
        }
        let node = &self.graph.node[*self.producer_index.get(value)?];
        (is_standard_onnx_domain(&node.domain) && node.output.len() == 1 && node.output[0] == value)
            .then_some(node)
    }

    fn single_producer_index(&self, value: &str) -> Option<usize> {
        self.single_producer(value)?;
        self.producer_index.get(value).copied()
    }

    /// Exact authored integer values of `value` (initializer or `Constant`
    /// node), restricted to the exact dtypes the folder evaluates in `i64`.
    /// Any anomaly — external payload, malformed union, float dtype — yields
    /// `None`, which leaves the original refusal in force.
    fn exact_integer_values(&self, value: &str) -> Option<Vec<i64>> {
        if self.initializer_count.get(value).copied() == Some(1)
            && !self.producer_index.contains_key(value)
        {
            let initializer = self
                .graph
                .initializer
                .iter()
                .find(|tensor| tensor.name == value)?;
            if !matches!(initializer.data_type, INT32 | INT64) {
                return None;
            }
            return exact_integer_initializer(initializer, None)
                .ok()
                .flatten()
                .map(|tensor| tensor.values);
        }
        let node = self.single_producer(value)?;
        if node.op_type != "Constant" {
            return None;
        }
        let tensor = exact_integer_constant(node, None).ok().flatten()?;
        matches!(tensor.dtype, INT32 | INT64).then_some(tensor.values)
    }

    /// Scalar (rank-0 or single-element) exact integer value.
    fn exact_scalar_integer(&self, value: &str) -> Option<i64> {
        let values = self.exact_integer_values(value)?;
        match values.as_slice() {
            [single] => Some(*single),
            _ => None,
        }
    }
}

struct CertifiedEqual {
    /// The `Shape`-node output operand: elementwise `>= 0` by ONNX semantics.
    shape_operand: String,
    negative_cone: Vec<usize>,
}

struct CertifiedSelect {
    pass_through: String,
    select_cone: Vec<usize>,
}

/// Certify `Equal(S, N)` (either operand order) where `S` is a `Shape` output
/// and `N` is provably elementwise negative with `shape(N) == shape(S)` (so
/// the all-false condition also has exactly `S`'s shape).
fn certify_equal(context: &CertifyContext<'_>, node: &NodeProto) -> Option<CertifiedEqual> {
    if !is_standard_onnx_domain(&node.domain)
        || node.op_type != "Equal"
        || node.input.len() != 2
        || node.input.iter().any(String::is_empty)
        || node.output.len() != 1
        || node.output[0].is_empty()
        || context.producer_count.get(node.output[0].as_str()).copied() != Some(1)
    {
        return None;
    }
    for (shape_side, negative_side) in [(0, 1), (1, 0)] {
        let shape_operand = &node.input[shape_side];
        if !is_shape_node_output(context, shape_operand) {
            continue;
        }
        if let Some(negative_cone) =
            certify_elementwise_negative(context, shape_operand, &node.input[negative_side])
        {
            return Some(CertifiedEqual {
                shape_operand: shape_operand.clone(),
                negative_cone,
            });
        }
    }
    None
}

fn is_shape_node_output(context: &CertifyContext<'_>, value: &str) -> bool {
    context.single_producer(value).is_some_and(|producer| {
        producer.op_type == "Shape" && producer.input.len() == 1 && !producer.input[0].is_empty()
    })
}

/// Prove `value` is elementwise strictly negative AND has exactly the shape of
/// `shape_operand`.  Two authored forms are admitted:
///
///  * a scalar/single-element negative integer constant (broadcast-neutral,
///    so the comparison keeps `shape_operand`'s shape), or
///  * `Mul(ConstantOfShape(Shape(shape_operand), fill=k>0), c)` with `c` a
///    scalar negative integer constant and `k*c` overflow-checked negative —
///    the authored cctsdb/ml4acopf `expand` idiom, whose VALUE is independent
///    of the runtime shape it is broadcast over and whose SHAPE is pinned to
///    `shape_operand` by its own `Shape` input.
fn certify_elementwise_negative(
    context: &CertifyContext<'_>,
    shape_operand: &str,
    value: &str,
) -> Option<Vec<usize>> {
    if let Some(scalar) = context.exact_scalar_integer(value) {
        if scalar < 0 {
            let mut cone = Vec::new();
            if let Some(index) = context.single_producer_index(value) {
                cone.push(index);
            }
            return Some(cone);
        }
        return None;
    }

    let mul = context.single_producer(value)?;
    if mul.op_type != "Mul" || mul.input.len() != 2 || mul.input.iter().any(String::is_empty) {
        return None;
    }
    for (ones_side, constant_side) in [(0, 1), (1, 0)] {
        let Some((fill, mut cone)) =
            certify_shape_anchored_fill(context, shape_operand, &mul.input[ones_side])
        else {
            continue;
        };
        let Some(constant) = context.exact_scalar_integer(&mul.input[constant_side]) else {
            continue;
        };
        // Overflow-checked: the product every lane holds must exist and be
        // strictly negative.
        let product = fill.checked_mul(constant)?;
        if fill <= 0 || constant >= 0 || product >= 0 {
            return None;
        }
        cone.push(context.single_producer_index(value)?);
        if let Some(index) = context.single_producer_index(&mul.input[constant_side]) {
            cone.push(index);
        }
        return Some(cone);
    }
    None
}

/// Certify `ConstantOfShape(Shape(shape_operand), fill)` and return the exact
/// positive-integer fill: a tensor of `fill` values whose shape is exactly
/// `shape(shape_operand)`.
fn certify_shape_anchored_fill(
    context: &CertifyContext<'_>,
    shape_operand: &str,
    value: &str,
) -> Option<(i64, Vec<usize>)> {
    let constant_of_shape = context.single_producer(value)?;
    if constant_of_shape.op_type != "ConstantOfShape"
        || constant_of_shape.input.len() != 1
        || constant_of_shape.input[0].is_empty()
    {
        return None;
    }
    let anchor = context.single_producer(&constant_of_shape.input[0])?;
    if anchor.op_type != "Shape" || anchor.input.len() != 1 || anchor.input[0] != shape_operand {
        return None;
    }
    // The optional `value` attribute must be an authored single-element exact
    // integer; the schema default (float32 zero) is neither.
    let fill = match constant_of_shape.attribute.as_slice() {
        [attribute] if attribute.name == "value" => {
            let tensor = attribute.t.as_ref()?;
            if !matches!(tensor.data_type, INT32 | INT64) {
                return None;
            }
            match exact_integer_initializer(tensor, None)
                .ok()
                .flatten()?
                .values
                .as_slice()
            {
                [single] => *single,
                _ => return None,
            }
        }
        _ => return None,
    };
    if fill <= 0 {
        return None;
    }
    let mut cone = vec![context.single_producer_index(value)?];
    if let Some(index) = context.single_producer_index(&constant_of_shape.input[0]) {
        cone.push(index);
    }
    Some((fill, cone))
}

/// Certify `Where(cond, X, Y)` as a value-identity of `Y`: `cond` is a
/// certified all-false `Equal` whose `Shape` operand IS `Y` (so `cond` and `Y`
/// share one shape and no lane selects `X`), and `X` cannot broaden the
/// broadcast (`X == Y`, or `X` is a `ConstantOfShape` anchored to `Shape(Y)`).
fn certify_where(
    context: &CertifyContext<'_>,
    certified_equals: &HashMap<String, String>,
    node: &NodeProto,
) -> Option<CertifiedSelect> {
    if !is_standard_onnx_domain(&node.domain)
        || node.op_type != "Where"
        || node.input.len() != 3
        || node.input.iter().any(String::is_empty)
        || node.output.len() != 1
        || node.output[0].is_empty()
        || context.producer_count.get(node.output[0].as_str()).copied() != Some(1)
    {
        return None;
    }
    let shape_operand = certified_equals.get(&node.input[0])?;
    let pass_through = &node.input[2];
    if pass_through != shape_operand {
        return None;
    }
    let select = &node.input[1];
    let select_cone = if select == pass_through {
        Vec::new()
    } else {
        let (_, cone) = certify_shape_anchored_fill(context, shape_operand, select)?;
        cone
    };
    Some(CertifiedSelect {
        pass_through: pass_through.clone(),
        select_cone,
    })
}

fn node_key(node: &NodeProto) -> NodeKey {
    (
        node.op_type.clone(),
        node.name.clone(),
        node.output.first().cloned().unwrap_or_default(),
    )
}

/// Discharge the deferred expand-identity refusals: rewire every consumer of
/// a certified `Where`'s output to the pass-through tensor `Y`, then delete
/// the certified `Where`s, `Equal`s and their private cones once nothing
/// consumes them.  Runs in `prepare_graph` BEFORE constant folding, on the
/// same authored graph the preflight certified.
///
/// Consumer rewiring is what keeps tensor-name continuity through the
/// canonicalization: the erased `Where` output name loses its producer, so
/// every reference to it must move to the name that keeps one.  A `Where`
/// whose output is an authored graph output is NOT erased (the output name
/// cannot move), so it survives and re-raises its deferred refusal.
///
/// Anything this pass does not erase re-raises its original schema refusal at
/// `LiteralExemptions::require_all_folded` — the admission cannot outlive its
/// certificate.
pub(super) fn canonicalize_expand_identity_cones(graph: &mut GraphProto) {
    let mut cones = ExpandIdentityCones::certify(graph);
    if cones.where_rewrites.is_empty() {
        return;
    }
    let graph_outputs: HashSet<&str> = graph
        .output
        .iter()
        .filter(|output| !output.name.is_empty())
        .map(|output| output.name.as_str())
        .collect();
    // Erased-output -> pass-through source.  Certified `Where` outputs are
    // unique (single-producer guard) and a pass-through is always a `Shape`
    // output, never another certified `Where` output, so one rewrite pass
    // cannot chain.
    let mut renames: HashMap<String, String> = HashMap::new();
    for rewrite in &cones.where_rewrites {
        let node = &graph.node[rewrite.node_index];
        if graph_outputs.contains(node.output[0].as_str()) {
            debug!(
                "expand-identity: Where '{}' is an authored graph output; leaving it to \
                 re-raise its deferred refusal",
                node.name
            );
            continue;
        }
        debug!(
            "expand-identity: Where '{}' has a certified all-false condition; rewiring \
             consumers of '{}' to pass-through '{}'",
            node.name, node.output[0], rewrite.pass_through
        );
        renames.insert(node.output[0].clone(), rewrite.pass_through.clone());
        cones.delete_candidates.insert(node_key(node));
    }
    for node in &mut graph.node {
        for input in &mut node.input {
            if let Some(pass_through) = renames.get(input.as_str()) {
                *input = pass_through.clone();
            }
        }
    }

    // Delete certified cone nodes that are now consumer-free.  Iterate to a
    // fixpoint: removing the rewired Where frees the Equal (and the select
    // cone), removing the Equal frees its Mul, then the ConstantOfShape, then
    // the inner Shape/Constant.  A cone node something else still reads
    // stays — and its schema refusal then re-raises, keeping this fail-closed.
    loop {
        let mut consumed: HashSet<&str> = graph
            .node
            .iter()
            .flat_map(|node| node.input.iter())
            .filter(|input| !input.is_empty())
            .map(String::as_str)
            .collect();
        consumed.extend(
            graph
                .output
                .iter()
                .filter(|output| !output.name.is_empty())
                .map(|output| output.name.as_str()),
        );
        let removable: Vec<String> = graph
            .node
            .iter()
            .filter(|node| {
                cones.delete_candidates.contains(&node_key(node))
                    && node
                        .output
                        .iter()
                        .filter(|output| !output.is_empty())
                        .all(|output| !consumed.contains(output.as_str()))
            })
            .map(|node| node.name.clone())
            .collect();
        if removable.is_empty() {
            break;
        }
        let removable: HashSet<String> = removable.into_iter().collect();
        graph.node.retain(|node| {
            !(removable.contains(&node.name) && cones.delete_candidates.contains(&node_key(node)))
        });
    }
}

/// ONNX `Cast` target dtypes whose drop is an exact identity.
const FLOAT32: i64 = 1;
const BOOL: i64 = 9;

/// ONNX operators SPECIFIED to return BOOL, so a `Cast(-> BOOL)` of their
/// output is a literal no-op.  Mirrors `convert::produces_boolean_valued_output`
/// (the predicate that makes `convert` drop the node in the first place); the
/// standard-domain guard is part of the guarantee.
fn produces_boolean_output(node: &NodeProto) -> bool {
    if !is_standard_onnx_domain(&node.domain) {
        return false;
    }
    matches!(
        node.op_type.as_str(),
        "Equal"
            | "Greater"
            | "GreaterOrEqual"
            | "Less"
            | "LessOrEqual"
            | "And"
            | "Or"
            | "Xor"
            | "Not"
            | "IsNaN"
            | "IsInf"
    ) || (node.op_type == "Cast" && cast_to_dtype(node) == Some(BOOL))
}

/// The `to` dtype of a standard-domain `Cast`, when authored as an INT.
fn cast_to_dtype(node: &NodeProto) -> Option<i64> {
    node.attribute
        .iter()
        .find(|attribute| attribute.name == "to")
        .filter(|attribute| attribute.r#type == crate::onnx_proto::attribute_type::INT)
        .map(onnx_int_attribute)
}

/// Erase exact-identity `Cast(-> FLOAT32)` nodes, rewiring their consumers to
/// the cast's operand.
///
/// WHY THIS EXISTS (#cctsdb-readout-continuity): `convert::node_to_layer`
/// already treats a FLOAT32-target cast as an exact identity and emits NO
/// layer for it ("FLOAT32 target: exact identity, all bound math is already
/// f32").  Dropping a node without forwarding its output name is only
/// invisible while nothing downstream reads that name — which was true for
/// every model that reached the builder before cctsdb's readout cone was
/// admitted.  patch-1 reads it four times (`Cast_199 -> Equal_200`,
/// `Cast_201 -> Cast_247`, `Cast_246`/`Cast_248 -> Mul_249`), so the builder
/// hit `activation input not found in tensor_to_node` and skipped live ops.
///
/// Erasing the node HERE, with consumer rewiring, is the same continuity
/// idiom [`canonicalize_expand_identity_cones`] uses: the surviving graph
/// reads only names that still have producers.  Value semantics are
/// unchanged — the drop already happened at conversion; this only makes the
/// name-forwarding explicit and early enough for const-folding, shape
/// inference and the builder to agree.
///
/// FAIL-CLOSED SCOPE: only `to = FLOAT32` casts in the standard domain, only
/// single-input/single-output nodes, and never a cast whose output is an
/// authored graph output (that name cannot move).  Integer targets keep their
/// guarded `Trunc` lowering, `BOOL` keeps its provable-identity branch, and
/// every unmodeled dtype keeps its refusal — all in `convert.rs`, untouched.
/// `fold_obligated_outputs`: outputs the schema preflight deferred to constant
/// folding (`LiteralExemptions`, `Discharge::Fold`). A Cast in that set MUST
/// survive to the folder: erasing it both strands the obligation (the
/// re-raise makes the model unloadable) and deletes the INT64->FLOAT dtype
/// conversion that launders integer provenance for the downstream float
/// folds (#qdq-crown-followup; collided live on smart_turn's
/// `Shape -> Slice -> Cast -> Sqrt` head-dim scaling under
/// `model.qdq_activation_relaxation`).
pub(super) fn canonicalize_identity_float_casts(
    graph: &mut GraphProto,
    fold_obligated_outputs: &HashSet<String>,
) {
    let graph_outputs: HashSet<&str> = graph
        .output
        .iter()
        .filter(|output| !output.name.is_empty())
        .map(|output| output.name.as_str())
        .collect();
    let producers: HashMap<&str, &NodeProto> = graph
        .node
        .iter()
        .flat_map(|node| {
            node.output
                .iter()
                .filter(|output| !output.is_empty())
                .map(move |output| (output.as_str(), node))
        })
        .collect();
    let mut renames: HashMap<String, String> = HashMap::new();
    let mut erased: HashSet<String> = HashSet::new();
    for node in &graph.node {
        if node.op_type != "Cast" || !is_standard_onnx_domain(&node.domain) {
            continue;
        }
        if node.input.len() != 1 || node.output.len() != 1 {
            continue;
        }
        if node.input[0].is_empty() || node.output[0].is_empty() {
            continue;
        }
        let identity = match cast_to_dtype(node) {
            // FLOAT32: every ny bound is already f32, so the cast is a no-op.
            Some(FLOAT32) => true,
            // BOOL: `x != 0`, an identity exactly on an already-BOOL operand.
            Some(BOOL) => producers
                .get(node.input[0].as_str())
                .is_some_and(|producer| produces_boolean_output(producer)),
            _ => false,
        };
        if !identity {
            continue;
        }
        if graph_outputs.contains(node.output[0].as_str()) {
            continue;
        }
        if fold_obligated_outputs.contains(node.output[0].as_str()) {
            debug!(
                "identity-cast: keeping Cast '{}' — its output '{}' carries a deferred-fold \
                 obligation, so constant folding (not erasure) must discharge it",
                node.name, node.output[0]
            );
            continue;
        }
        debug!(
            "identity-cast: erasing exact-identity Cast '{}' and rewiring consumers of '{}' to '{}'",
            node.name, node.output[0], node.input[0]
        );
        renames.insert(node.output[0].clone(), node.input[0].clone());
        erased.insert(node.name.clone());
    }
    if renames.is_empty() {
        return;
    }
    // Chains (`Cast -> Cast`) resolve to the first surviving producer.
    for node in &mut graph.node {
        for input in &mut node.input {
            let mut hops = 0;
            while let Some(source) = renames.get(input.as_str()) {
                let source = source.clone();
                *input = source;
                hops += 1;
                if hops > renames.len() {
                    break;
                }
            }
        }
    }
    graph
        .node
        .retain(|node| !(erased.contains(&node.name) && renames.contains_key(&node.output[0])));
}

/// Read an INT attribute's value across the proto's accessor spelling.
fn onnx_int_attribute(attribute: &crate::onnx_proto::AttributeProto) -> i64 {
    attribute.i_value()
}
