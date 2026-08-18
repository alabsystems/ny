// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unified IBP dispatch for graph nodes.
//!
//! Extracts the repeated match-on-layer-type dispatch pattern shared across
//! 5 IBP propagation methods into a single source of truth. Each caller
//! provides its own input resolution strategy via a closure; the dispatch
//! logic here handles layer arity detection, Concat constant_inputs
//! interleaving, and propagation method selection.
//!
//! Design: `designs/2026-03-03-graph-ibp-dispatch-dedup.md`
//! Issues: #2405, #1948, #1856

use crate::layers::{BoundPropagation, ConcatLayer, Layer};

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::super::GraphNode;

/// Classified input names for a graph node, determined by layer arity.
///
/// Borrow-preserving classifier: returns borrowed input *names* (not resolved
/// tensors) so callers can map names to bounds using their own resolution
/// strategy — owned via closure, or borrowed via `bounds_ref()`.
///
/// Shared by [`resolve_node_inputs`] and `graph_alpha/bounds/ibp.rs` to avoid
/// duplicating the same arity/first-input classification logic.
///
/// Design: Slice B of `designs/2026-03-10-issue-2633-graphnode-accessor-refresh.md`
pub(crate) enum ResolvedInputNames<'a> {
    /// Single input (most layers, SkipMerge, OpaqueSkip,
    /// Where with embedded constants).
    Unary(&'a str),
    /// Two inputs (Add, Sub, Mul, Div, MatMul, Min, Max, etc.).
    Binary(&'a str, &'a str),
    /// Three inputs (Where without embedded constants, SelfAttention).
    Ternary(&'a str, &'a str, &'a str),
    /// Variable inputs (Concat). `dynamic_inputs` contains names for non-constant
    /// graph edges; `has_constants` indicates that `ConcatLayer::constant_inputs`
    /// must be consulted for interleaving.
    NaryConcat {
        dynamic_inputs: &'a [String],
        has_constants: bool,
    },
}

fn classify_variable_arity_node_inputs<'a>(
    node: &'a GraphNode,
    node_name: &str,
    op_name: &str,
    activation_input_count: usize,
) -> Result<ResolvedInputNames<'a>> {
    match activation_input_count {
        1 => {
            let input_name = node.require_unary_input().map_err(|_| {
                NyError::InvalidSpec(format!(
                    "{op_name} node {node_name} requires 1 input, got {}",
                    node.inputs.len()
                ))
            })?;
            Ok(ResolvedInputNames::Unary(input_name))
        }
        2 => {
            let (input_a, input_b) = node.require_binary_inputs().map_err(|_| {
                NyError::InvalidSpec(format!(
                    "{op_name} node {node_name} requires 2 inputs, got {}",
                    node.inputs.len()
                ))
            })?;
            Ok(ResolvedInputNames::Binary(input_a, input_b))
        }
        3 => {
            let (input_a, input_b, input_c) = node.require_ternary_inputs().map_err(|_| {
                NyError::InvalidSpec(format!(
                    "{op_name} node {node_name} requires 3 inputs, got {}",
                    node.inputs.len()
                ))
            })?;
            Ok(ResolvedInputNames::Ternary(input_a, input_b, input_c))
        }
        other => Err(NyError::InvalidSpec(format!(
            "{op_name} node {node_name} has invalid activation arity {other}"
        ))),
    }
}

/// Classify a graph node's input arity and return borrowed input names.
///
/// Uses `GraphNode` safe accessors (`require_unary_input`, `require_binary_inputs`,
/// `require_ternary_inputs`) instead of direct `node.inputs[N]` indexing. Returns
/// structured names that callers map to bounds using their own resolution strategy.
///
/// **Soundness note:** Concat is checked *before* `is_binary()` because
/// `Layer::is_binary()` returns true for Concat. Without this ordering,
/// n-ary Concat (3+ inputs) would silently drop inputs beyond the first two.
/// See #2405.
pub(crate) fn classify_node_inputs<'a>(
    node: &'a GraphNode,
    node_name: &str,
) -> Result<ResolvedInputNames<'a>> {
    match &node.layer {
        Layer::Where(w) => {
            if w.has_embedded_constants() {
                let cond_input = node.require_unary_input().map_err(|_| {
                    NyError::InvalidSpec(format!(
                        "Where node {} with embedded constants requires 1 input (condition)",
                        node_name
                    ))
                })?;
                Ok(ResolvedInputNames::Unary(cond_input))
            } else {
                let (cond, x, y) = node.require_ternary_inputs().map_err(|_| {
                    NyError::InvalidSpec(format!("Where node {} requires 3 inputs", node_name))
                })?;
                Ok(ResolvedInputNames::Ternary(cond, x, y))
            }
        }
        Layer::ScatterNd(scatter) => classify_variable_arity_node_inputs(
            node,
            node_name,
            "ScatterND",
            scatter.activation_input_count(),
        ),
        Layer::ScatterAdd(scatter) => classify_variable_arity_node_inputs(
            node,
            node_name,
            "ScatterAdd",
            scatter.activation_input_count(),
        ),
        Layer::IndexAdd(index) => classify_variable_arity_node_inputs(
            node,
            node_name,
            "IndexAdd",
            index.activation_input_count(),
        ),
        Layer::SelfAttention(_) => {
            let (query_input, key_input, value_input) =
                node.require_ternary_inputs().map_err(|_| {
                    NyError::InvalidSpec(format!(
                        "SelfAttention node {} requires 3 inputs, got {}",
                        node_name,
                        node.inputs.len()
                    ))
                })?;
            Ok(ResolvedInputNames::Ternary(
                query_input,
                key_input,
                value_input,
            ))
        }
        // Variable-style AdaIN: (x, style_gamma, style_beta).
        // Fixed-style AdaIN falls through to the unary catch-all.
        Layer::AdaIN1d(adain) if adain.requires_style_inputs() => {
            let (x_input, ny_input, beta_input) =
                node.require_ternary_inputs().map_err(|_| {
                    NyError::InvalidSpec(format!(
                        "Variable-style AdaIN1d node {} requires 3 inputs (x, style_gamma, style_beta), got {}",
                        node_name,
                        node.inputs.len()
                    ))
                })?;
            Ok(ResolvedInputNames::Ternary(x_input, ny_input, beta_input))
        }
        Layer::SkipMerge(_) => {
            if node.inputs.len() != 1 {
                return Err(NyError::InvalidSpec(format!(
                    "SkipMerge node {} expects exactly 1 input, got {}. \
                     Use OpaqueSkip for multi-input skipped ops.",
                    node_name,
                    node.inputs.len()
                )));
            }
            let input_name = node.require_unary_input().map_err(|_| {
                NyError::InvalidSpec(format!(
                    "SkipMerge node {} expects exactly 1 input, got {}. \
                     Use OpaqueSkip for multi-input skipped ops.",
                    node_name,
                    node.inputs.len()
                ))
            })?;
            Ok(ResolvedInputNames::Unary(input_name))
        }
        Layer::OpaqueSkip(_) => {
            // OpaqueSkip is intentionally opaque: it ignores its inputs' values
            // and emits conservative unbounded bounds shaped like its first input.
            // Multi-input skipped ops are legal (#2666), so accept >=1 inputs and
            // resolve only the first for shape — do NOT require exactly one input.
            let input_name = node.inputs.first().map(String::as_str).ok_or_else(|| {
                NyError::InvalidSpec(format!("OpaqueSkip node {} has no inputs", node_name))
            })?;
            Ok(ResolvedInputNames::Unary(input_name))
        }
        // Concat MUST be matched before is_binary() — see soundness note above.
        Layer::Concat(concat) => {
            let has_constants = concat.constant_inputs.is_some();
            Ok(ResolvedInputNames::NaryConcat {
                dynamic_inputs: &node.inputs,
                has_constants,
            })
        }
        _ if node.layer.is_binary() => {
            let (input_a, input_b) = node.require_binary_inputs().map_err(|_| {
                NyError::InvalidSpec(format!("Binary node {} requires 2 inputs", node_name))
            })?;
            Ok(ResolvedInputNames::Binary(input_a, input_b))
        }
        _ => {
            let input_name = node
                .require_unary_input()
                .map_err(|_| NyError::InvalidSpec(format!("Node {} has no inputs", node_name)))?;
            Ok(ResolvedInputNames::Unary(input_name))
        }
    }
}

/// Resolved inputs for a graph node, determined by layer arity.
///
/// Produced by [`resolve_node_inputs`] and consumed by [`dispatch_ibp_resolved`].
pub(crate) enum ResolvedInputs {
    /// Unary: single input bounds (most layers, SkipMerge, OpaqueSkip,
    /// Where with embedded constants).
    Unary(BoundedTensor),
    /// Binary: two input bounds (Add, Sub, Mul, Div, MatMul, Min, Max, etc.).
    Binary(BoundedTensor, BoundedTensor),
    /// Ternary: three input bounds (Where without embedded constants,
    /// SelfAttention). Third element boxed to reduce enum size (clippy::large_enum_variant).
    Ternary(BoundedTensor, BoundedTensor, Box<BoundedTensor>),
    /// N-ary: variable number of inputs (Concat with constant_inputs
    /// reconstruction).
    Nary(Vec<BoundedTensor>),
}

/// Resolve inputs for a graph node using the provided resolver closure.
///
/// The `resolve` closure maps an input node name to a `BoundedTensor`. Callers
/// provide their own resolution strategy:
/// - `bounds_ref()` + clone for standard IBP
/// - `concrete_value()` + `BoundedTensor::concrete()` for statistics
/// - `bounds_for_block()` for block-wise IBP
///
/// The layer type determines input arity: Where (1 or 3), SelfAttention (3),
/// SkipMerge (1), OpaqueSkip (1), Concat (n-ary with constant interleaving),
/// binary ops (2), everything else (1).
///
/// **Soundness note:** Concat is checked *before* `is_binary()` because
/// `Layer::is_binary()` returns true for Concat. Without this ordering,
/// n-ary Concat (3+ inputs) would silently drop inputs beyond the first two.
/// See #2405.
pub(crate) fn resolve_node_inputs<F>(
    node: &GraphNode,
    node_name: &str,
    resolve: &mut F,
) -> Result<ResolvedInputs>
where
    F: FnMut(&str) -> Result<BoundedTensor>,
{
    // Single source of truth: classify arity via classify_node_inputs, then
    // resolve each name to a BoundedTensor. This eliminates the duplicated
    // match block that previously mirrored classify_node_inputs.
    // Design: designs/2026-03-13-ibp-dispatch-classify-then-resolve.md
    let classified = classify_node_inputs(node, node_name)?;
    match classified {
        ResolvedInputNames::Unary(name) => Ok(ResolvedInputs::Unary(resolve(name)?)),
        ResolvedInputNames::Binary(a, b) => Ok(ResolvedInputs::Binary(resolve(a)?, resolve(b)?)),
        ResolvedInputNames::Ternary(a, b, c) => Ok(ResolvedInputs::Ternary(
            resolve(a)?,
            resolve(b)?,
            Box::new(resolve(c)?),
        )),
        ResolvedInputNames::NaryConcat { .. } => {
            let concat = match &node.layer {
                Layer::Concat(c) => c,
                _ => {
                    return Err(NyError::InternalError(format!(
                        "classify_node_inputs returned NaryConcat for non-Concat node '{}'",
                        node_name
                    )))
                }
            };
            let inputs = resolve_concat_inputs(node, node_name, concat, resolve)?;
            Ok(ResolvedInputs::Nary(inputs))
        }
    }
}

/// Resolve Concat inputs, interleaving graph edges and embedded constants.
///
/// When `constant_inputs` is present, the vec is indexed by original ONNX input
/// order: `Some(tensor)` entries are constants embedded at graph construction
/// time, `None` entries are dynamic graph edges resolved via the closure.
/// `node.inputs` only contains names for the dynamic (non-constant) inputs.
fn resolve_concat_inputs<F>(
    node: &GraphNode,
    node_name: &str,
    concat: &ConcatLayer,
    resolve: &mut F,
) -> Result<Vec<BoundedTensor>>
where
    F: FnMut(&str) -> Result<BoundedTensor>,
{
    if let Some(ref ci) = concat.constant_inputs {
        let mut graph_idx = 0;
        ci.iter()
            .map(|const_opt| {
                if let Some(constant) = const_opt {
                    Ok(constant.clone())
                } else {
                    let name = node.inputs.get(graph_idx).ok_or_else(|| {
                        NyError::InternalError(format!(
                            "Concat '{}': ran out of graph inputs at graph_idx {}",
                            node_name, graph_idx
                        ))
                    })?;
                    graph_idx += 1;
                    resolve(name)
                }
            })
            .collect()
    } else {
        node.inputs.iter().map(|name| resolve(name)).collect()
    }
}

/// Propagate IBP bounds through already-resolved inputs.
///
/// This is the second half of the dispatch: given [`ResolvedInputs`] from
/// [`resolve_node_inputs`], invoke the appropriate propagation method on the
/// node's layer.
///
/// Callers that need custom handling for specific input arities (e.g., zonotope
/// tightening for binary MatMul ops) should use [`resolve_node_inputs`] directly
/// and handle the relevant [`ResolvedInputs`] variant themselves, delegating
/// the rest to this function.
pub(crate) fn dispatch_ibp_resolved(
    node: &GraphNode,
    node_name: &str,
    inputs: ResolvedInputs,
) -> Result<BoundedTensor> {
    match inputs {
        ResolvedInputs::Unary(input) => match &node.layer {
            Layer::Where(w) if w.has_embedded_constants() => w.propagate_ibp_with_condition(&input),
            Layer::ScatterAdd(scatter) => scatter.propagate_ibp(&input),
            Layer::IndexAdd(index) => index.propagate_ibp(&input),
            Layer::ScatterNd(scatter) => scatter.propagate_ibp(&input),
            _ => node.layer.propagate_ibp(&input),
        },
        ResolvedInputs::Binary(a, b) => match &node.layer {
            Layer::ScatterAdd(scatter) => scatter.propagate_ibp_binary(&a, &b),
            Layer::IndexAdd(index) => index.propagate_ibp_binary(&a, &b),
            Layer::ScatterNd(scatter) => scatter.propagate_ibp_binary(&a, &b),
            _ => node.layer.propagate_ibp_binary(&a, &b),
        },
        ResolvedInputs::Ternary(a, b, c) => match &node.layer {
            Layer::Where(w) => w.propagate_ibp_ternary(&a, &b, &c),
            Layer::SelfAttention(attn) => attn.propagate_ibp_ternary(&a, &b, &c),
            Layer::ScatterAdd(scatter) => scatter.propagate_ibp_ternary(&a, &b, &c),
            Layer::IndexAdd(index) => index.propagate_ibp_ternary(&a, &b, &c),
            Layer::ScatterNd(scatter) => scatter.propagate_ibp_ternary(&a, &b, &c),
            // Variable-style AdaIN: (x, style_gamma, style_beta).
            Layer::AdaIN1d(adain) => adain.propagate_ibp_ternary(&a, &b, &c),
            _ => Err(NyError::InternalError(format!(
                "Unexpected ternary dispatch for node '{}'",
                node_name
            ))),
        },
        ResolvedInputs::Nary(inputs) => match &node.layer {
            Layer::Concat(concat) => {
                let refs: Vec<&BoundedTensor> = inputs.iter().collect();
                concat.propagate_ibp_nary(&refs)
            }
            _ => Err(NyError::InternalError(format!(
                "Unexpected n-ary dispatch for node '{}'",
                node_name
            ))),
        },
    }
}

/// Resolve inputs and propagate IBP bounds through a single node.
///
/// Convenience function combining [`resolve_node_inputs`] and
/// [`dispatch_ibp_resolved`]. Use this for callers with no custom binary-op
/// handling (e.g., `propagate_ibp_with_clipper`, `collect_activation_statistics`).
///
/// For callers that need zonotope tightening on binary ops, call
/// [`resolve_node_inputs`] directly, handle the `Binary` case with tightening,
/// and delegate remaining cases to [`dispatch_ibp_resolved`].
pub(crate) fn dispatch_ibp_for_node<F>(
    node: &GraphNode,
    node_name: &str,
    resolve: &mut F,
) -> Result<BoundedTensor>
where
    F: FnMut(&str) -> Result<BoundedTensor>,
{
    let inputs = resolve_node_inputs(node, node_name, resolve)?;
    dispatch_ibp_resolved(node, node_name, inputs)
}

/// NaN firewall: check output bounds for NaN and return `NumericalInstability`
/// error if any element is NaN.
///
/// Single source of truth for NaN policy across all IBP paths (#2706, #2812).
/// NaN bounds corrupt all downstream nodes and are never valid — this is the
/// consistent error behavior established by the NaN Strategy design
/// (`designs/archive/2026-02-25-nan-strategy-unification.md`).
///
/// `context` identifies the calling path for diagnostics (e.g., "IBP detailed").
pub(crate) fn check_nan_firewall(
    bounds: &BoundedTensor,
    context: &str,
    node_name: &str,
    layer_type: &str,
) -> Result<()> {
    let has_nan = bounds
        .lower()
        .iter()
        .chain(bounds.upper().iter())
        .any(|v| v.is_nan());
    if has_nan {
        return Err(NyError::NumericalInstability(format!(
            "{}: NaN bounds at node '{}' ({})",
            context, node_name, layer_type
        )));
    }
    Ok(())
}

/// Pollable counterpart to [`check_nan_firewall`].
///
/// The NaN policy and diagnostic are identical. A poll error aborts the scan
/// before the bounds can be cached by a deadline-authoritative caller.
pub(crate) fn check_nan_firewall_with_poll<F>(
    bounds: &BoundedTensor,
    context: &str,
    node_name: &str,
    layer_type: &str,
    mut poll: F,
) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    const POLL_ELEMENTS: usize = 4_096;

    poll()?;
    for (index, value) in bounds
        .lower()
        .iter()
        .chain(bounds.upper().iter())
        .enumerate()
    {
        if index.is_multiple_of(POLL_ELEMENTS) {
            poll()?;
        }
        if value.is_nan() {
            return Err(NyError::NumericalInstability(format!(
                "{}: NaN bounds at node '{}' ({})",
                context, node_name, layer_type
            )));
        }
    }
    poll()
}

/// Keep the per-element tighter of a zonotope result and the plain IBP result
/// for the same graph node.
///
/// Both inputs are sound over-approximations of the identical node, so their
/// element-wise intersection (max of lowers, min of uppers) is also sound and
/// never looser than either operand. This guards the SwiGLU `MulBinary`
/// zonotope path: on kernels with a large (e.g. RMSNorm-induced) base width the
/// zonotope's coarse scale-normalized quadratic multiply can be *looser* than
/// plain IBP, so intersecting avoids a regression while still capturing the
/// up/gate correlation gains wherever the zonotope wins.
///
/// `intersection_per_element` widens to the union on the numerically-degenerate
/// disjoint/NaN cases (still sound); a `None` (shape mismatch — not expected
/// for two views of one node) keeps the zonotope result, matching the prior
/// unconditional-zonotope behavior.
pub(crate) fn intersect_zonotope_ibp(zono: BoundedTensor, ibp: BoundedTensor) -> BoundedTensor {
    match zono.intersection_per_element(&ibp) {
        Some((tightened, _disjoint)) => tightened,
        None => zono,
    }
}

/// Pollable counterpart to [`intersect_zonotope_ibp`].
///
/// A poll error prevents publication of a partially constructed intersection.
pub(crate) fn intersect_zonotope_ibp_with_poll<F>(
    zono: BoundedTensor,
    ibp: BoundedTensor,
    poll: F,
) -> Result<BoundedTensor>
where
    F: FnMut() -> Result<()>,
{
    match zono.intersection_per_element_with_poll(&ibp, poll)? {
        Some((tightened, _disjoint)) => Ok(tightened),
        None => Ok(zono),
    }
}
