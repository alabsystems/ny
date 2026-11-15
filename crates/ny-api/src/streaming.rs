// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Streaming / memory-efficient verification of large networks.
//!
//! Re-exports the curated streaming verification surface used by external
//! consumers to verify large transformer networks under tight memory budgets.
//! [`StreamingVerifier`] applies gradient checkpointing during CROWN
//! propagation: instead of retaining bounds at every layer (O(L*N) memory), it
//! stores [`CheckpointedBounds`] at intervals controlled by [`StreamingConfig`]
//! and recomputes forward on the backward pass, trading compute for memory.
//! Use [`estimate_memory_savings`] to size the trade-off before verifying.
//!
//! ## Graph entry points
//!
//! [`StreamingVerifier`]'s own entries are **sequential-only**: every one of
//! them takes a [`SequentialNetwork`], because the checkpoint store indexes
//! bounds by layer position and the backward pass walks that position list in
//! reverse. Downstream producers (traced models in particular) hand ny a
//! [`GraphNetwork`], so this module provides the graph entry points as a
//! **sound bridge** rather than a second implementation:
//!
//! * A graph that IS a chain — a single-path DAG: every node consumes exactly
//!   one input, the walk from the output reaches [`NETWORK_INPUT`], and that
//!   walk covers *every* node in the graph — is converted to the equivalent
//!   [`SequentialNetwork`] by [`chain_graph_to_sequential`] and streamed
//!   through the sequential verifier.
//! * Any other graph **fails closed** with a precise
//!   [`NyError::UnsupportedOp`] naming the offending node and the reason. There
//!   is deliberately NO silent fallback to non-streaming propagation: a caller
//!   that asked for a memory-bounded run must be told its network cannot have
//!   one, not handed an O(L*N) run that may exhaust the budget it was
//!   protecting.
//!
//! ### Why the conversion is faithful — and where it is not
//!
//! Being a chain is **not** sufficient. Copying `node.layer()` into a
//! [`SequentialNetwork`] is faithful only for layer kinds where the graph node
//! dispatcher and the sequential streaming verifier make the *same per-layer
//! calls*. They do not for every kind:
//!
//! * **Node-context shape injection.** The graph dispatcher clones the layer
//!   and stamps this node's live input shape into it
//!   (`set_input_shape` / `set_input_length`) before propagating — see
//!   `network/backward_dispatch/dispatch.rs`. The sequential streaming verifier
//!   has no node context and propagates whatever shape the layer *value*
//!   happens to carry. When that carried shape is stale but has the same
//!   element count, nothing errors: the backward pass simply builds a
//!   **different linear map**, and intersecting it with the (correct) forward
//!   IBP box yields a bound that is tight and WRONG. This was a real false
//!   proof for `Conv2d` (see `conv2d_chain_graph_must_refuse_not_return_a_wrong_bound`).
//! * **Certified vs. plain forward kernels.** For the conv family and
//!   `AveragePool` the graph forward calls the layer's *certified*
//!   `propagate_ibp_sound_*` kernel; the sequential streaming forward calls the
//!   plain `propagate_ibp`, which under cancellation can be **narrower** than
//!   the certified enclosure.
//! * **Embedded constants / reduced arity.** `Concat`, `Where`,
//!   `ScatterND` / `ScatterAdd` / `IndexAdd`, `SelfAttention` and variable-style
//!   `AdaIN1d` can present a single graph edge while carrying their remaining
//!   operands inside the layer, and are dispatched through layer-specific
//!   graph-only entries.
//! * **Graph-only node semantics.** `SkipMerge` / `OpaqueSkip` pass through by
//!   *node input*, and `OpaqueSkip` taints downstream nodes so that
//!   index-producing ops (`ArgMax`, `Topk`, `NonZero`, ...) degrade to
//!   declared-shape unbounded bounds. A flat layer list carries none of that.
//! * **`IntervalConst`** must be read as a rigorous enclosure through the
//!   certified coefficient-error channel; the streaming verifier's generic
//!   backward arm does not carry that channel.
//!
//! [`sequential_bridge_refusal`] therefore fails closed on all of the above.
//!
//! ### Compile-time exhaustiveness (the reason this is a `match`, not a list)
//!
//! The refusal decision is an **exhaustive `match &Layer` with no catch-all
//! arm**, mirroring `network/backward_dispatch/dispatch.rs` and
//! `layers/layer_enum/dispatch.rs`. Adding a variant to `ny_propagate`'s
//! `Layer` is therefore a **compile error here** until someone makes an
//! explicit allow/refuse decision for it. The previous string allow-by-default
//! list silently admitted every unnamed variant — the exact opposite of the
//! fail-closed posture its own doc comment promised — and admitted `Conv2d`,
//! `ConvTranspose2d`, `Transpose`, `Tile` and `Gather` in the process.
//! **Never reintroduce a `_ =>` arm.**
//!
//! ### Known residual gap (NOT closed by this bridge)
//!
//! With soundness rounding engaged, the graph IBP driver widens *every* node's
//! box outward (`in_features + 2` ULPs for `Linear`, 1 ULP elsewhere) after the
//! layer call. The sequential streaming forward
//! (`StreamingVerifier::collect_checkpointed_bounds`) applies no such widening,
//! so a streamed box can be narrower than the certified graph box by that
//! rounding term. This is a **driver-level policy difference that affects every
//! kind uniformly** — it is a property of `ny-propagate`'s sequential streaming
//! forward, identical for a hand-built [`SequentialNetwork`], and not something
//! the bridge can close by refusing layer kinds. The per-kind refusals above
//! cover the cases where the *layer call itself* differs.

use std::collections::HashSet;

use ny_core::{GemmEngine, NyError, Result};
use ny_propagate::layers::Layer;
use ny_propagate::{GraphNetwork, Network as SequentialNetwork, NETWORK_INPUT};
use ny_tensor::BoundedTensor;

pub use ny_propagate::{
    estimate_memory_savings, CheckpointedBounds, StreamingConfig, StreamingVerifier,
};

/// The graph node dispatcher rewrites this layer with the node's live input
/// shape before propagating; a sequential layer call cannot.
const SHAPE_INJECTED: &str =
    "the graph node dispatcher stamps this node's live input shape into the layer \
     (set_input_shape/set_input_length) before propagating, while the sequential streaming \
     verifier propagates whatever shape the layer value carries — a stale shape with a matching \
     element count silently builds a DIFFERENT linear map instead of erroring";

/// The graph forward runs the layer's certified sound IBP kernel; the
/// sequential streaming forward runs the plain one.
const CERTIFIED_FORWARD: &str =
    "the graph forward calls this layer's CERTIFIED propagate_ibp_sound_* kernel while the \
     sequential streaming forward calls the plain propagate_ibp, whose result can be narrower \
     than the certified enclosure under cancellation";

/// Multi-input in the graph even when a single edge is present.
const EMBEDDED_OPERANDS: &str =
    "it is dispatched through a graph-only multi-input entry and can carry its remaining operands \
     embedded in the layer, so a single graph edge does not make it unary";

/// Semantics that live on the graph node, not on the layer.
const GRAPH_NODE_SEMANTICS: &str =
    "its meaning is defined by graph node structure (input pass-through selection and downstream \
     taint propagation), which a flat sequential layer list does not carry";

/// Index-producing ops the graph degrades under OpaqueSkip taint.
const SHAPE_AS_VALUE: &str =
    "it reads its input's SHAPE as part of its output VALUE, and the graph degrades it to \
     declared-shape unbounded bounds when tainted by an upstream OpaqueSkip — a node-level \
     decision the sequential streaming verifier never makes";

/// Why `layer` cannot be carried across the graph -> sequential streaming
/// bridge, or `None` when its sequential interpretation is verified identical
/// to its graph interpretation.
///
/// # Exhaustive by construction
///
/// [`bridge_refusal_reason`] is an exhaustive `match` with **no catch-all**, so
/// a new `Layer` variant fails to compile here until someone decides whether the
/// bridge can carry it. See the module docs.
fn sequential_bridge_refusal(layer: &Layer) -> Option<String> {
    let kind = layer.layer_type();
    // Arity first: it yields the most specific message for the variable-arity
    // kinds (ScatterND/ScatterAdd/IndexAdd/AdaIN1d), which the exhaustive match
    // below refuses unconditionally anyway.
    if layer.is_binary() {
        return Some(format!(
            "layer '{kind}' is a binary operation (two bounded inputs); the sequential streaming \
             verifier propagates one input per layer"
        ));
    }
    if layer.is_ternary() {
        return Some(format!(
            "layer '{kind}' is a ternary operation (three bounded inputs); the sequential \
             streaming verifier propagates one input per layer"
        ));
    }
    let reason = bridge_refusal_reason(layer)?;
    Some(format!(
        "layer '{kind}' is not carried across the graph -> sequential streaming bridge: {reason}"
    ))
}

/// The allow-list itself: `None` allows `layer` across the bridge, `Some(reason)`
/// refuses it.
///
/// A kind is allowed ONLY when the sequential streaming verifier makes the same
/// per-layer calls the graph node dispatcher makes — `Layer::propagate_ibp`
/// forward, and `Layer::propagate_crown_backward` (or the identical
/// `propagate_linear_with_engine` arm for `Linear`) backward — with no node
/// context injected in between. Anything else is refused: a conservative
/// refusal is always sound, a wrong translation never is.
///
/// # Do not add a `_ =>` arm
///
/// The `match` is exhaustive on purpose. A newly added `Layer` variant MUST be
/// classified here explicitly; silence would widen what the bridge accepts, and
/// widening it wrongly is a false proof.
#[allow(clippy::match_same_arms)] // the arms document distinct refusal reasons
fn bridge_refusal_reason(layer: &Layer) -> Option<&'static str> {
    match layer {
        Layer::PointLinear(_) => Some(
            "diagnostic point-only original-weight storage lacks a qualified interval or CROWN proof-bound adapter",
        ),
        // ================= ALLOWED =================
        // Graph: `l.propagate_linear_with_engine_and_deadline(...)`.
        // Streaming: `l.propagate_linear_with_engine(...)`. Identical with no
        // deadline; the deadline only aborts, it never changes the map.
        Layer::Linear(_) => None,

        // Elementwise activations. Shape-independent, dispatched through the
        // graph's "remaining unary layers: unified trait dispatch" arm, which is
        // literally `layer.propagate_crown_backward(node_lb, Some(pre))` — the
        // same call the streaming verifier's generic arm makes. (Graph ReLU may
        // additionally use an optimized alpha; that is a tighter SOUND
        // relaxation of the SAME function, not a different network.)
        Layer::ReLU(_)
        | Layer::LeakyReLU(_)
        | Layer::Clip(_)
        | Layer::Elu(_)
        | Layer::Selu(_)
        | Layer::PRelu(_)
        | Layer::HardSigmoid(_)
        | Layer::HardSwish(_)
        | Layer::SiLU(_)
        | Layer::Exp(_)
        | Layer::Log(_)
        | Layer::Celu(_)
        | Layer::Mish(_)
        | Layer::ThresholdedRelu(_)
        | Layer::Shrink(_)
        | Layer::Softsign(_)
        | Layer::Snake(_)
        | Layer::GELU(_)
        | Layer::Tanh(_)
        | Layer::Sigmoid(_)
        | Layer::Erf(_)
        | Layer::Softplus(_)
        | Layer::Sin(_)
        | Layer::Cos(_)
        | Layer::Tan(_)
        | Layer::Arctan(_)
        | Layer::Abs(_)
        | Layer::Sqrt(_)
        | Layer::Reciprocal(_)
        | Layer::PowConstant(_)
        | Layer::Floor(_)
        | Layer::Ceil(_)
        | Layer::Round(_)
        | Layer::Trunc(_)
        | Layer::Sign(_)
        | Layer::Compare(_) => None,

        // Unary arithmetic against a constant baked into the layer. The
        // constant travels WITH the layer value, so the sequential copy holds
        // the same operand the graph node holds.
        Layer::AddConstant(_)
        | Layer::SubConstant(_)
        | Layer::MulConstant(_)
        | Layer::DivConstant(_) => None,

        // Elementwise outward widening; no node context.
        Layer::QdqPerturbation(_) | Layer::EpsilonExpand(_) => None,

        // Softmax family: unified trait dispatch on both sides. (The graph's
        // `PreserveLeadingAxis` forward arm is a per-position MODE that the
        // sequential `Network` has too; the streaming verifier's plain
        // `Layer::propagate_ibp` is the `Plain` mode on both sides.)
        Layer::Softmax(_)
        | Layer::CausalSoftmax(_)
        | Layer::LogSoftmax(_)
        | Layer::LogSumExp(_) => None,

        // Normalization: unified trait dispatch on both sides.
        Layer::LayerNorm(_)
        | Layer::InstanceNorm1d(_)
        | Layer::GroupNorm(_)
        | Layer::BatchNorm(_) => None,

        // RmsNorm: the graph has its own arm only to thread a GenBaB `inv_rms`
        // override. With no override it is byte-identical to the generic
        // `propagate_crown_backward` path, and an override only NARROWS the
        // graph's certified interval — so the streamed bound is never the
        // tighter of the two.
        Layer::RmsNorm(_) => None,

        // Variable-style AdaIN1d takes (x, style_gamma, style_beta) and is
        // dispatched as a ternary CROWN in the graph. `is_ternary()` already
        // catches it; refuse by name too so the decision is visible here.
        Layer::AdaIN1d(adain) if adain.requires_style_inputs() => Some(
            "variable-style AdaIN1d consumes (x, style_gamma, style_beta) and is dispatched as a \
             ternary graph CROWN; only the fixed-style (style embedded in the layer) form is unary",
        ),
        // Fixed-style AdaIN1d: style embedded in the layer, unified trait dispatch.
        Layer::AdaIN1d(_) => None,

        // Pure shape metadata. Backward is an identity/validation on the
        // coefficient matrix, and the target shape travels with the layer.
        Layer::Reshape(_) | Layer::Flatten(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => None,

        // Exact-linear structural ops dispatched through the unified trait arm
        // on BOTH sides — the graph injects no node context for these.
        Layer::Pad(_) | Layer::Resize(_) | Layer::RoPE(_) => None,

        // Reductions dispatched through the unified trait arm on both sides.
        Layer::ReduceMean(_)
        | Layer::ReduceSum(_)
        | Layer::ReduceMax(_)
        | Layer::ReduceMin(_)
        | Layer::CumSum(_) => None,

        // MaxPool2d: unified trait dispatch on both sides. (The VGG MaxPool
        // decomposition is an explicit graph REWRITE that replaces the node, so
        // a surviving MaxPool2d node is dispatched plainly.)
        Layer::MaxPool2d(_) => None,

        // ================= REFUSED =================
        // The confirmed false proof. `Conv2d`/`ConvTranspose2d` get
        // `set_input_shape(H, W)` and `Conv1d`/`ConvTranspose1d` get
        // `set_input_length(L)` from the node's live pre-activation in the graph
        // dispatcher; the streaming verifier does not (and for the 2D transpose
        // does not even reach an engine-aware arm). On top of that the graph
        // forward uses the certified sound kernel.
        Layer::Conv1d(_)
        | Layer::Conv2d(_)
        | Layer::ConvTranspose1d(_)
        | Layer::ConvTranspose2d(_) => Some(
            "the graph node dispatcher stamps this node's live input shape into the convolution \
             (set_input_shape/set_input_length) before propagating and runs its CERTIFIED \
             propagate_ibp_sound_* forward, while the sequential streaming verifier propagates \
             whatever shape the layer value carries and runs the plain propagate_ibp — a stale \
             shape with a matching element count silently builds a DIFFERENT linear map instead \
             of erroring",
        ),

        // Conv3d is IBP-ONLY by construction: its CROWN backward fails closed
        // with `UnsupportedOp` because no certified rank-3-spatial transposed
        // convolution exists yet (see `Layer::Conv3d`'s own doc). The bridge's
        // admission test is that BOTH the forward and the
        // `propagate_crown_backward` call match the graph dispatcher's, so a
        // layer with no backward at all can never be carried. Refused on its
        // own arm rather than folded into the shape-injection group, because
        // the blocking reason is different and outlives it.
        Layer::Conv3d(_) => Some(
            "it is IBP-only: its CROWN backward fails closed with UnsupportedOp because no \
             certified rank-3-spatial transposed convolution exists, so the sequential streaming \
             verifier's backward call has nothing sound to dispatch to",
        ),

        // Same shape-injection hazard, no conv arithmetic.
        Layer::Transpose(_) | Layer::Tile(_) | Layer::Slice(_) => Some(SHAPE_INJECTED),

        // Shape injection AND indices that may be embedded in the layer.
        Layer::Gather(_) => Some(
            "the graph node dispatcher stamps this node's live input shape into the Gather \
             (set_input_shape) before propagating — and the runtime shape-query form reads that \
             shape as its output VALUE — while the sequential streaming verifier propagates \
             whatever shape the layer value carries",
        ),

        // Certified sound forward kernel only on the graph side.
        Layer::AveragePool(_) => Some(CERTIFIED_FORWARD),

        // The frozen constant is a rigorous ENCLOSURE that must be discharged
        // through the certified coefficient-error channel. The graph routes it
        // to a dedicated arm; the streaming verifier's generic backward does not
        // carry that channel, and an enclosure silently read as its midpoint is
        // exactly the unsound point-fold this carrier exists to prevent.
        Layer::IntervalConst(_) => Some(
            "its frozen constant is a rigorous interval ENCLOSURE that must be discharged through \
             the certified coefficient-error channel, which the sequential streaming backward does \
             not carry — reading the enclosure as its midpoint is an unsound point fold",
        ),

        // Multi-input in the graph even behind a single edge.
        Layer::Concat(_)
        | Layer::Where(_)
        | Layer::SelfAttention(_)
        | Layer::ScatterNd(_)
        | Layer::ScatterAdd(_)
        | Layer::IndexAdd(_) => Some(EMBEDDED_OPERANDS),

        // Node-structural semantics.
        Layer::SkipMerge(_) | Layer::OpaqueSkip(_) => Some(GRAPH_NODE_SEMANTICS),

        // Index producers under the graph's taint-gated shape-as-value degrade.
        Layer::Topk(_)
        | Layer::ArgMax(_)
        | Layer::ArgMin(_)
        | Layer::ArgSort(_)
        | Layer::NonZero(_) => Some(SHAPE_AS_VALUE),

        // Binary operations: two bounded inputs. `is_binary()` reports these
        // first with an arity-specific message; listed here so the match stays
        // exhaustive and the decision is explicit.
        Layer::MatMul(_)
        | Layer::MulBinary(_)
        | Layer::Add(_)
        | Layer::Sub(_)
        | Layer::Div(_)
        | Layer::Atan2(_)
        | Layer::BilinearCrown(_)
        | Layer::MinBinary(_)
        | Layer::MaxBinary(_)
        | Layer::ExpandLikeLastAxis(_)
        | Layer::CompareTensor(_) => Some(
            "it is a binary operation (two bounded inputs); the sequential streaming verifier \
             propagates one input per layer",
        ),
        // NOTE: no `_ =>` arm. See the doc comment above.
    }
}

/// Convert a chain [`GraphNetwork`] into the equivalent [`SequentialNetwork`],
/// or fail closed explaining exactly why the graph is not a chain.
///
/// A graph qualifies as a chain when ALL of the following hold:
///
/// 1. it has an output node;
/// 2. every node on the walk from the output node back toward the input
///    consumes exactly one input;
/// 3. every such node's layer kind is on the bridge's exhaustive allow-list —
///    a genuine unary layer whose sequential streaming interpretation is
///    verified identical to its graph interpretation (see
///    [`sequential_bridge_refusal`]). Being a chain is NOT enough on its own:
///    `Conv2d`, `Transpose`, `Gather` and friends present a single graph edge
///    and are still refused, because the graph dispatcher rewrites them with
///    node context the sequential verifier cannot supply;
/// 4. the walk terminates at [`NETWORK_INPUT`]; and
/// 5. the walk visits **every** node the graph holds — no orphan node, no side
///    branch, and (because each visited node is distinct and every node is
///    visited) no node consumed by two successors.
///
/// The returned network's layers are in input -> output order, so
/// `network.layers()[i]` is the `i`-th operation the graph applies.
///
/// # Errors
///
/// [`NyError::UnsupportedOp`] naming the offending node and the violated
/// condition. This is the ONLY outcome for a non-chain graph: the bridge never
/// degrades to a non-streaming path (see the module docs).
pub fn chain_graph_to_sequential(graph: &GraphNetwork) -> Result<SequentialNetwork> {
    let output = graph.output_name();
    if output.is_empty() {
        return Err(NyError::UnsupportedOp(
            "streaming graph entry: the graph has no output node set".to_string(),
        ));
    }
    let total_nodes = graph.num_nodes();

    // Walk output -> input, collecting layers in reverse order.
    let mut reversed: Vec<Layer> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut current = output.to_string();
    loop {
        if !visited.insert(current.clone()) {
            return Err(NyError::UnsupportedOp(format!(
                "streaming graph entry: node '{current}' is revisited — the graph is cyclic, not \
                 a chain"
            )));
        }
        let node = graph.node(&current).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "streaming graph entry: dangling node reference '{current}'"
            ))
        })?;
        let layer = node.layer();
        if let Some(reason) = sequential_bridge_refusal(layer) {
            return Err(NyError::UnsupportedOp(format!(
                "streaming graph entry: node '{current}' cannot stream — {reason}. Streaming is \
                 available only for chain (single-path) graphs; run this network through the \
                 regular graph verifier instead."
            )));
        }
        let inputs = node.inputs();
        if inputs.len() != 1 {
            return Err(NyError::UnsupportedOp(format!(
                "streaming graph entry: node '{current}' has {} inputs — streaming requires a \
                 chain (single-path) graph where every node consumes exactly one input",
                inputs.len()
            )));
        }
        reversed.push(layer.clone());
        let predecessor = inputs[0].clone();
        if predecessor == NETWORK_INPUT {
            break;
        }
        current = predecessor;
    }

    // Coverage: a chain's walk must account for every node in the graph. A
    // shorter walk means the graph carries a side branch, a second input-fed
    // node, or an orphan — none of which a linear layer list can represent.
    if visited.len() != total_nodes {
        return Err(NyError::UnsupportedOp(format!(
            "streaming graph entry: the walk from output '{output}' to the network input covers \
             {} of the graph's {total_nodes} node(s) — streaming requires a chain (single-path) \
             graph with no side branches or orphan nodes",
            visited.len()
        )));
    }

    reversed.reverse();
    let mut network = SequentialNetwork::new();
    for layer in reversed {
        network.add_layer(layer);
    }
    Ok(network)
}

/// [`StreamingVerifier::collect_checkpointed_bounds`] for a chain
/// [`GraphNetwork`].
///
/// # Errors
/// Fails closed via [`chain_graph_to_sequential`] when `graph` is not a chain,
/// then propagates any propagation error.
pub fn collect_checkpointed_bounds_graph(
    verifier: &StreamingVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> Result<CheckpointedBounds> {
    let network = chain_graph_to_sequential(graph)?;
    verifier.collect_checkpointed_bounds(&network, input)
}

/// [`StreamingVerifier::propagate_crown_streaming`] for a chain
/// [`GraphNetwork`].
///
/// # Errors
/// Fails closed via [`chain_graph_to_sequential`] when `graph` is not a chain,
/// then propagates any propagation error.
pub fn propagate_crown_streaming_graph(
    verifier: &StreamingVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> Result<BoundedTensor> {
    let network = chain_graph_to_sequential(graph)?;
    verifier.propagate_crown_streaming(&network, input)
}

/// [`StreamingVerifier::propagate_crown_streaming_with_engine`] for a chain
/// [`GraphNetwork`].
///
/// # Errors
/// Fails closed via [`chain_graph_to_sequential`] when `graph` is not a chain,
/// then propagates any propagation error.
pub fn propagate_crown_streaming_graph_with_engine(
    verifier: &StreamingVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    let network = chain_graph_to_sequential(graph)?;
    verifier.propagate_crown_streaming_with_engine(&network, input, engine)
}

/// [`StreamingVerifier::propagate_crown_batched_streaming`] for a chain
/// [`GraphNetwork`].
///
/// # Errors
/// Fails closed via [`chain_graph_to_sequential`] when `graph` is not a chain,
/// then propagates any propagation error.
pub fn propagate_crown_batched_streaming_graph(
    verifier: &StreamingVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> Result<BoundedTensor> {
    let network = chain_graph_to_sequential(graph)?;
    verifier.propagate_crown_batched_streaming(&network, input)
}

/// [`StreamingVerifier::propagate_crown_batched_streaming_with_engine`] for a
/// chain [`GraphNetwork`].
///
/// # Errors
/// Fails closed via [`chain_graph_to_sequential`] when `graph` is not a chain,
/// then propagates any propagation error.
pub fn propagate_crown_batched_streaming_graph_with_engine(
    verifier: &StreamingVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    let network = chain_graph_to_sequential(graph)?;
    verifier.propagate_crown_batched_streaming_with_engine(&network, input, engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use ndarray::{Array1, Array2};
    use ny_propagate::GraphNode;

    fn linear_layer(weight: Vec<f32>, rows: usize, cols: usize, bias: Vec<f32>) -> Layer {
        let w = Array2::from_shape_vec((rows, cols), weight).expect("weight shape");
        Layer::Linear(LinearLayer::new(w, Some(Array1::from(bias))).expect("valid linear"))
    }

    /// `input -> linear1 -> relu -> linear2` — a genuine chain.
    fn chain_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            linear_layer(vec![1.0, -1.0, -1.0, 1.0], 2, 2, vec![0.0, 0.0]),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            linear_layer(vec![1.0, 1.0, 1.0, -1.0], 2, 2, vec![0.5, -0.5]),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");
        graph
    }

    /// The same three layers as `chain_graph`, built directly as a sequential
    /// network — the reference the bridge must reproduce.
    fn chain_sequential() -> SequentialNetwork {
        let mut net = SequentialNetwork::new();
        net.add_layer(linear_layer(
            vec![1.0, -1.0, -1.0, 1.0],
            2,
            2,
            vec![0.0, 0.0],
        ));
        net.add_layer(Layer::ReLU(ReLULayer::new()));
        net.add_layer(linear_layer(
            vec![1.0, 1.0, 1.0, -1.0],
            2,
            2,
            vec![0.5, -0.5],
        ));
        net
    }

    /// Diamond: `input -> {a, b} -> add` — NOT a chain (binary fan-in).
    fn diamond_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "a",
            linear_layer(vec![1.0, 0.0, 0.0, 1.0], 2, 2, vec![0.0, 0.0]),
        ));
        graph.add_node(GraphNode::from_input(
            "b",
            linear_layer(vec![0.0, 1.0, 1.0, 0.0], 2, 2, vec![0.0, 0.0]),
        ));
        graph.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["a".to_string(), "b".to_string()],
        ));
        graph.set_output("add");
        graph
    }

    /// Two independent input-fed nodes, output on one of them — the walk covers
    /// one node but the graph holds two (orphan / side branch).
    fn orphan_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "kept",
            linear_layer(vec![1.0, 0.0, 0.0, 1.0], 2, 2, vec![0.0, 0.0]),
        ));
        graph.add_node(GraphNode::from_input(
            "orphan",
            linear_layer(vec![2.0, 0.0, 0.0, 2.0], 2, 2, vec![0.0, 0.0]),
        ));
        graph.set_output("kept");
        graph
    }

    fn unit_input() -> BoundedTensor {
        BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32, -1.0]).into_dyn(),
            ndarray::arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("valid input box")
    }

    #[test]
    fn chain_graph_converts_to_the_same_layer_sequence() {
        let network = chain_graph_to_sequential(&chain_graph()).expect("chain converts");
        let reference = chain_sequential();
        assert_eq!(
            network.num_layers(),
            reference.num_layers(),
            "converted chain must have one layer per graph node"
        );
        let got: Vec<&str> = network.layers().iter().map(Layer::layer_type).collect();
        let want: Vec<&str> = reference.layers().iter().map(Layer::layer_type).collect();
        assert_eq!(
            got, want,
            "converted layers must be in input->output order: {want:?}"
        );
    }

    #[test]
    fn streaming_graph_entry_matches_the_sequential_entry() {
        let verifier = StreamingVerifier::new(StreamingConfig {
            checkpoint_interval: 1,
            ..Default::default()
        });
        let input = unit_input();

        let via_graph = propagate_crown_streaming_graph(&verifier, &chain_graph(), &input)
            .expect("graph streaming runs");
        let via_sequential = verifier
            .propagate_crown_streaming(&chain_sequential(), &input)
            .expect("sequential streaming runs");

        assert_eq!(
            via_graph.lower(),
            via_sequential.lower(),
            "graph entry must reproduce the sequential lower bounds exactly"
        );
        assert_eq!(
            via_graph.upper(),
            via_sequential.upper(),
            "graph entry must reproduce the sequential upper bounds exactly"
        );
    }

    #[test]
    fn streaming_graph_bounds_enclose_concrete_evaluations() {
        // Soundness spot-check: the streamed enclosure must contain the exact
        // output at the box corners.
        let verifier = StreamingVerifier::new(StreamingConfig::default());
        let bounds = propagate_crown_streaming_graph(&verifier, &chain_graph(), &unit_input())
            .expect("graph streaming runs");

        for (x0, x1) in [(-1.0_f32, -1.0_f32), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            // linear1: (x0 - x1, x1 - x0); relu; linear2: (h0 + h1 + 0.5, h0 - h1 - 0.5)
            let h0 = (x0 - x1).max(0.0);
            let h1 = (x1 - x0).max(0.0);
            let y = [h0 + h1 + 0.5, h0 - h1 - 0.5];
            for (i, expected) in y.iter().enumerate() {
                let lo = bounds.lower()[i];
                let hi = bounds.upper()[i];
                assert!(
                    lo <= *expected && *expected <= hi,
                    "streamed bounds [{lo}, {hi}] must enclose exact output {expected} at \
                     ({x0}, {x1}) for output {i}"
                );
            }
        }
    }

    #[test]
    fn batched_streaming_graph_entry_matches_the_sequential_entry() {
        let verifier = StreamingVerifier::new(StreamingConfig::default());
        let input = unit_input();

        let via_graph = propagate_crown_batched_streaming_graph(&verifier, &chain_graph(), &input)
            .expect("graph batched streaming runs");
        let via_sequential = verifier
            .propagate_crown_batched_streaming(&chain_sequential(), &input)
            .expect("sequential batched streaming runs");

        assert_eq!(via_graph.lower(), via_sequential.lower());
        assert_eq!(via_graph.upper(), via_sequential.upper());
    }

    #[test]
    fn checkpoint_collection_graph_entry_matches_the_sequential_entry() {
        let verifier = StreamingVerifier::new(StreamingConfig {
            checkpoint_interval: 1,
            ..Default::default()
        });
        let input = unit_input();

        let via_graph = collect_checkpointed_bounds_graph(&verifier, &chain_graph(), &input)
            .expect("graph checkpoint collection runs");
        let via_sequential = verifier
            .collect_checkpointed_bounds(&chain_sequential(), &input)
            .expect("sequential checkpoint collection runs");

        assert_eq!(
            via_graph.num_checkpoints(),
            via_sequential.num_checkpoints(),
            "graph entry must produce the same checkpoint schedule"
        );
    }

    #[test]
    fn non_chain_graph_fails_closed_with_a_precise_message() {
        let err = propagate_crown_streaming_graph(
            &StreamingVerifier::new(StreamingConfig::default()),
            &diamond_graph(),
            &unit_input(),
        )
        .expect_err("a binary fan-in graph must NOT stream");
        let message = err.to_string();
        assert!(
            matches!(err, NyError::UnsupportedOp(_)),
            "non-chain graphs must fail closed as UnsupportedOp, got {err:?}"
        );
        assert!(
            message.contains("'add'"),
            "the refusal must name the offending node: {message}"
        );
        assert!(
            message.contains("chain"),
            "the refusal must state the chain requirement: {message}"
        );
    }

    #[test]
    fn orphan_node_graph_fails_closed_on_coverage() {
        let err = chain_graph_to_sequential(&orphan_graph())
            .expect_err("a graph with an off-path node must NOT convert");
        let message = err.to_string();
        assert!(matches!(err, NyError::UnsupportedOp(_)), "{err:?}");
        assert!(
            message.contains("covers 1 of the graph's 2 node(s)"),
            "the refusal must report the coverage shortfall: {message}"
        );
    }

    #[test]
    fn missing_output_fails_closed() {
        let graph = GraphNetwork::new();
        let err = chain_graph_to_sequential(&graph).expect_err("no output node");
        assert!(matches!(err, NyError::UnsupportedOp(_)), "{err:?}");
        assert!(err.to_string().contains("no output node"));
    }

    #[test]
    fn every_graph_only_dispatch_layer_is_refused_by_name() {
        // Pin the embedded-operand / reduced-arity refusals: each of these can
        // present a single graph edge and must still not cross the bridge.
        let concat = Layer::Concat(crate::layers::ConcatLayer::new(0));
        let refusal = sequential_bridge_refusal(&concat)
            .expect("Concat must be refused by the sequential bridge");
        assert!(
            refusal.contains("Concat"),
            "the refusal must name the layer: {refusal}"
        );

        let attention = Layer::SelfAttention(crate::layers::SelfAttentionLayer::standard());
        let refusal = sequential_bridge_refusal(&attention)
            .expect("SelfAttention must be refused by the sequential bridge");
        assert!(
            refusal.contains("SelfAttention"),
            "the refusal must name the layer: {refusal}"
        );

        let where_layer = Layer::Where(crate::layers::WhereLayer::new());
        let refusal = sequential_bridge_refusal(&where_layer)
            .expect("Where must be refused by the sequential bridge");
        assert!(
            refusal.contains("Where"),
            "the refusal must name the layer: {refusal}"
        );
    }

    // ---------------------------------------------------------------------
    // #false-proof: the reviewer probe, and the per-kind refusals that close it.
    // ---------------------------------------------------------------------

    /// The reviewer probe, verbatim in effect.
    ///
    /// `input -> conv` is a textbook CHAIN: one node, one input, the walk
    /// reaches `NETWORK_INPUT` and covers every node. Before the fix the bridge
    /// copied the `Conv2d` layer into a `SequentialNetwork` and streamed it.
    ///
    /// The conv is a 1x1 kernel with `stride = (2, 1)`, so it SELECTS spatial
    /// rows 0, 2, ... — *which* input elements those are depends entirely on the
    /// spatial shape the CROWN backward believes the input has. The graph
    /// dispatcher stamps the live `(4, 3)` shape in; the layer value carries a
    /// stale `(2, 6)` (same 12 elements, so nothing errors). The two disagree:
    ///
    /// * true / graph output element 3 = `x[6]`, whose box is `[5, 25]`;
    /// * the stale-shape backward computes element 3 = `x[3]`, box `[0, 15]`.
    ///
    /// `tighten_crown_output` then intersects the WRONG backward with the right
    /// forward box and publishes `[5, 15]` — tight, and a FALSE PROOF: `25` is
    /// attained at a concrete input inside the box. (Measured before the fix:
    /// `[5.0, 15.000001]` vs the graph's `[5.0, 25.0]`.)
    ///
    /// The bridge must now REFUSE rather than answer.
    fn conv2d_stale_shape_chain_graph() -> GraphNetwork {
        use ny_propagate::layers::Conv2dLayer;
        let kernel = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 1, 1]), vec![1.0_f32])
            .expect("kernel shape");
        let mut conv = Conv2dLayer::new(kernel, None, (2, 1), (0, 0)).expect("valid conv2d");
        conv.set_input_shape(2, 6);
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.set_output("conv");
        graph
    }

    /// The probe's input box: element 3 in `[0, 15]`, element 6 in `[5, 25]`,
    /// every other element pinned to 0.
    fn conv2d_probe_input() -> BoundedTensor {
        let mut lower = vec![0.0_f32; 12];
        let mut upper = vec![0.0_f32; 12];
        upper[3] = 15.0;
        lower[6] = 5.0;
        upper[6] = 25.0;
        BoundedTensor::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 3]), lower).expect("lower"),
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 3]), upper).expect("upper"),
        )
        .expect("valid input box")
    }

    #[test]
    fn conv2d_chain_graph_must_refuse_not_return_a_wrong_bound() {
        let graph = conv2d_stale_shape_chain_graph();
        let input = conv2d_probe_input();
        let verifier = StreamingVerifier::new(StreamingConfig::default());

        // The graph verifier's answer for output element (0, 1, 0): the true
        // range [5, 25], with 25 attained at the concrete input x[6] = 25.
        let reference = graph
            .propagate_ibp(&input)
            .expect("the graph propagates this chain fine");
        assert_eq!(
            (reference.lower()[[0, 1, 0]], reference.upper()[[0, 1, 0]]),
            (5.0_f32, 25.0_f32),
            "probe precondition: the graph's own answer must be [5, 25]"
        );

        for (entry, result) in [
            (
                "propagate_crown_streaming_graph",
                propagate_crown_streaming_graph(&verifier, &graph, &input),
            ),
            (
                "propagate_crown_batched_streaming_graph",
                propagate_crown_batched_streaming_graph(&verifier, &graph, &input),
            ),
            (
                "collect_checkpointed_bounds_graph",
                collect_checkpointed_bounds_graph(&verifier, &graph, &input).map(|_| input.clone()),
            ),
        ] {
            let err = result.err().unwrap_or_else(|| {
                panic!("{entry} must REFUSE a Conv2d chain, not answer for a different network")
            });
            assert!(
                matches!(err, NyError::UnsupportedOp(_)),
                "{entry}: the refusal must be UnsupportedOp, got {err:?}"
            );
            let message = err.to_string();
            assert!(
                message.contains("Conv2d"),
                "{entry}: the refusal must name the layer: {message}"
            );
            assert!(
                message.contains("'conv'"),
                "{entry}: the refusal must name the offending node: {message}"
            );
        }
    }

    /// One refusal per newly-refused kind. Each of these presents a SINGLE graph
    /// edge, so the chain walk alone would have admitted it; only the exhaustive
    /// allow-list stops them.
    #[test]
    fn newly_refused_kinds_fail_closed_by_name() {
        use ny_propagate::layers::{
            Conv2dLayer, ConvTranspose2dLayer, GatherLayer, TileLayer, TransposeLayer,
        };

        let conv_kernel = || {
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 1, 1]), vec![1.0_f32])
                .expect("kernel shape")
        };
        let cases: Vec<(&str, Layer)> = vec![
            (
                "Conv2d",
                Layer::Conv2d(
                    Conv2dLayer::new(conv_kernel(), None, (1, 1), (0, 0)).expect("valid conv2d"),
                ),
            ),
            (
                "ConvTranspose2d",
                Layer::ConvTranspose2d(
                    ConvTranspose2dLayer::new(conv_kernel(), None, (1, 1), (0, 0))
                        .expect("valid conv transpose 2d"),
                ),
            ),
            (
                "Transpose",
                Layer::Transpose(TransposeLayer::new(vec![1, 0])),
            ),
            ("Tile", Layer::Tile(TileLayer::new(0, 2))),
            ("Gather", Layer::Gather(GatherLayer::new(0, None, vec![1]))),
        ];

        for (kind, layer) in cases {
            // 1. The refusal helper itself names the layer.
            let refusal = sequential_bridge_refusal(&layer).unwrap_or_else(|| {
                panic!("{kind} must be refused by the sequential streaming bridge")
            });
            assert!(
                refusal.contains(kind),
                "the {kind} refusal must name the layer: {refusal}"
            );

            // 2. And a single-edge chain graph carrying it fails closed, rather
            //    than converting and streaming a differently-interpreted net.
            let mut graph = GraphNetwork::new();
            graph.add_node(GraphNode::from_input("node", layer));
            graph.set_output("node");
            let err = match chain_graph_to_sequential(&graph) {
                Ok(_) => panic!("a chain graph carrying {kind} must NOT convert"),
                Err(err) => err,
            };
            assert!(
                matches!(err, NyError::UnsupportedOp(_)),
                "{kind}: must fail closed as UnsupportedOp, got {err:?}"
            );
            let message = err.to_string();
            assert!(
                message.contains(kind) && message.contains("'node'"),
                "the {kind} refusal must name the layer and the node: {message}"
            );
        }
    }

    /// The allow-list is a decision, not an accident: the kinds the bridge is
    /// verified to carry must stay carried, and a spot-check of the refused
    /// classes must stay refused. This is the runtime companion to the
    /// compile-time exhaustiveness of `bridge_refusal_reason` — the `match` has
    /// no `_ =>` arm, so a new `Layer` variant cannot slip in silently.
    #[test]
    fn allow_list_carries_exactly_the_verified_kinds() {
        use ny_propagate::layers::{
            AveragePoolLayer, IntervalConstLayer, OpaqueSkipLayer, ReshapeLayer,
        };

        let allowed: Vec<Layer> = vec![
            linear_layer(vec![1.0, 0.0, 0.0, 1.0], 2, 2, vec![0.0, 0.0]),
            Layer::ReLU(ReLULayer::new()),
            Layer::Sigmoid(crate::layers::SigmoidLayer::new()),
            Layer::Reshape(ReshapeLayer::new(vec![4])),
        ];
        for layer in &allowed {
            assert!(
                sequential_bridge_refusal(layer).is_none(),
                "{} must stay on the streaming bridge allow-list",
                layer.layer_type()
            );
        }

        let interval_center = LinearLayer::new(
            Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("center"),
            None,
        )
        .expect("valid center");
        let refused: Vec<Layer> = vec![
            Layer::AveragePool(AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false)),
            Layer::IntervalConst(
                IntervalConstLayer::try_new_linear(
                    interval_center,
                    Array2::from_shape_vec((1, 1), vec![0.5_f32]).expect("radius"),
                )
                .expect("valid interval const"),
            ),
            Layer::OpaqueSkip(OpaqueSkipLayer::new()),
        ];
        for layer in &refused {
            let kind = layer.layer_type();
            let refusal = sequential_bridge_refusal(layer)
                .unwrap_or_else(|| panic!("{kind} must be refused by the streaming bridge"));
            assert!(
                refusal.contains(kind),
                "the {kind} refusal must name the layer: {refusal}"
            );
        }
    }

    #[test]
    fn diagnostic_point_only_storage_refuses_streaming_bridge_and_public_proof_admission() {
        use ny_propagate::certify::work::WorkBudget;
        use ny_propagate::layers::{PointLinearBuildPolicy, PointLinearLayer};
        use std::time::Duration;

        fn refused<T>(result: Result<T>) {
            match result {
                Err(NyError::UnsupportedOp(message)) => {
                    assert!(message.contains("PointLinear") && message.contains("point-only"));
                }
                Err(error) => panic!("diagnostic proof admission requires UnsupportedOp: {error}"),
                Ok(_) => panic!("diagnostic storage must not acquire streaming proof authority"),
            }
        }
        let budget = WorkBudget::new(None, 1 << 20, 1 << 20, 1 << 20, 16, Duration::ZERO)
            .expect("compact fixture budget");
        let weights = Array2::eye(2);
        let compact = Layer::PointLinear(
            PointLinearLayer::from_original_view(
                weights.view(),
                false,
                None,
                PointLinearBuildPolicy {
                    budget: &budget,
                    deadline: None,
                    max_resident_bytes: 1 << 20,
                    max_scratch_bytes: 1 << 20,
                    max_work: 1 << 20,
                },
            )
            .expect("original-weight diagnostic fixture"),
        );
        let reason = bridge_refusal_reason(&compact).expect("compact has no proof-bound adapter");
        assert!(reason.contains("diagnostic point-only") && reason.contains("proof-bound adapter"));
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("diagnostic", compact));
        graph.set_output("diagnostic");
        refused(graph.ensure_proof_eligible());
        refused(chain_graph_to_sequential(&graph));
        let verifier = StreamingVerifier::new(StreamingConfig {
            checkpoint_interval: 1,
            ..Default::default()
        });
        refused(propagate_crown_streaming_graph(
            &verifier,
            &graph,
            &unit_input(),
        ));

        let ordinary = linear_layer(vec![1.0, 0.0, 0.0, 1.0], 2, 2, vec![0.0, 0.0]);
        assert!(bridge_refusal_reason(&ordinary).is_none());
        let mut control = GraphNetwork::new();
        control.add_node(GraphNode::from_input("ordinary", ordinary));
        control.set_output("ordinary");
        assert!(control.ensure_proof_eligible().is_ok());
        assert!(chain_graph_to_sequential(&control).is_ok());
        let bounds = propagate_crown_streaming_graph(&verifier, &control, &unit_input())
            .expect("ordinary Linear still streams");
        let input = unit_input();
        assert!(bounds
            .lower()
            .iter()
            .zip(input.lower())
            .all(|(&lo, &x)| lo <= x));
        assert!(bounds
            .upper()
            .iter()
            .zip(input.upper())
            .all(|(&hi, &x)| hi >= x));
    }

    /// The positive half of the contract: a genuinely-supported chain still
    /// streams, and the streamed enclosure agrees with what the graph verifier
    /// computes for the SAME network — same layer sequence, and an enclosure
    /// that contains the graph's own answer nowhere-tighter than the truth.
    #[test]
    fn supported_chain_streams_and_agrees_with_the_graph_verifier() {
        let graph = chain_graph();
        let input = unit_input();
        let verifier = StreamingVerifier::new(StreamingConfig {
            checkpoint_interval: 1,
            ..Default::default()
        });

        let streamed = propagate_crown_streaming_graph(&verifier, &graph, &input)
            .expect("a Linear/ReLU/Linear chain must still stream");

        // Same network: the graph verifier's CROWN answer on the same graph.
        let graph_bounds = ny_propagate::Verifier::new(ny_propagate::PropagationConfig {
            method: ny_propagate::PropagationMethod::Crown,
            ..Default::default()
        })
        .certify_graph_bounds("streaming-bridge-parity", &graph, &input, None)
        .expect("graph certification runs");
        let graph_bounds = match graph_bounds {
            ny_propagate::composition::certificate::BoundCertificationResult::Certified(cert) => {
                cert.output_bounds().clone()
            }
            other => panic!("graph certification must complete, got {other:?}"),
        };

        assert_eq!(
            streamed.shape(),
            graph_bounds.shape(),
            "the streamed answer must be for the same network shape as the graph verifier's"
        );
        for i in 0..streamed.len() {
            let (slo, shi) = (streamed.lower()[i], streamed.upper()[i]);
            let (glo, ghi) = (graph_bounds.lower()[i], graph_bounds.upper()[i]);
            // Bit-identical is the expectation for this chain (the graph and the
            // streaming verifier make the same per-layer calls here), but assert
            // agreement to a rounding tolerance so a legitimate ULP-scale
            // difference in either driver does not read as a wrong network.
            assert!(
                (slo - glo).abs() <= 1e-5 && (shi - ghi).abs() <= 1e-5,
                "output {i}: streamed [{slo}, {shi}] must match the graph verifier's \
                 [{glo}, {ghi}] — a large gap means the bridge streamed a DIFFERENT network"
            );
        }

        // And the streamed enclosure must still contain every concrete output.
        for (x0, x1) in [(-1.0_f32, -1.0_f32), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            let h0 = (x0 - x1).max(0.0);
            let h1 = (x1 - x0).max(0.0);
            for (i, expected) in [h0 + h1 + 0.5, h0 - h1 - 0.5].iter().enumerate() {
                let (lo, hi) = (streamed.lower()[i], streamed.upper()[i]);
                assert!(
                    lo <= *expected && *expected <= hi,
                    "streamed bounds [{lo}, {hi}] must enclose exact output {expected} at \
                     ({x0}, {x1}) for output {i}"
                );
            }
        }
    }
}
