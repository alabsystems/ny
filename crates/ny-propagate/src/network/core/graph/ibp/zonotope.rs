// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Zonotope propagation helpers for graph networks.

use crate::layers::{Layer, LayerNormMode};

use ny_core::{NyError, Result};
use ny_tensor::{BoundedTensor, ZonotopeTensor};
use tracing::{debug, info};

use super::super::{GraphNetwork, GraphNode, NETWORK_INPUT};
use super::dispatch::{check_nan_firewall, dispatch_ibp_for_node};
use super::zonotope_matmul::propagate_disjoint_matmul;

/// How graph zonotope propagation should handle Softmax-family nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ZonotopeSoftmaxMode {
    /// Use the zonotope affine relaxation for Softmax-family nodes.
    #[default]
    Affine,
    /// Intentionally cut the zonotope pipeline at Softmax-family nodes and
    /// reuse the existing graph-level IBP fallback for that operator.
    IntervalFallback,
}

/// Operator-local controls for graph zonotope propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub struct ZonotopePropagationOptions {
    softmax_mode: ZonotopeSoftmaxMode,
}

impl ZonotopePropagationOptions {
    /// Default graph zonotope options.
    pub const fn new() -> Self {
        Self {
            softmax_mode: ZonotopeSoftmaxMode::Affine,
        }
    }

    /// Softmax-family handling mode for this propagation run.
    pub const fn softmax_mode(self) -> ZonotopeSoftmaxMode {
        self.softmax_mode
    }

    /// Override Softmax-family handling for this propagation run.
    pub const fn with_softmax_mode(mut self, softmax_mode: ZonotopeSoftmaxMode) -> Self {
        self.softmax_mode = softmax_mode;
        self
    }
}

impl Default for ZonotopePropagationOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphNetwork {
    /// Propagate bounds through the graph using zonotopes for correlation-aware attention.
    ///
    /// **This is a DIAGNOSTIC / localization instrument, not a production
    /// tightening path, and it has no production callers by design** (#orphans-lies-1).
    /// Its consumers are the stage-by-stage width-localization harnesses
    /// (`ny-onnx` Whisper `attention_stage_localization` / `attention_context_matrix`
    /// / `attention_context_fallback`) that ask "which operator is the widener?"
    /// by running IBP and this pass over the same prefix and printing the ratio.
    /// [`ZonotopeSoftmaxMode::IntervalFallback`] exists for exactly that bisection.
    ///
    /// # Why it is not routed
    ///
    /// The obvious reading — that the graph IBP paths are correlation-blind and
    /// this is the tightening they are missing — is wrong on all three counts:
    ///
    /// 1. **The correlation win is already in production, node-locally.**
    ///    `propagate_ibp` and its three siblings run
    ///    `try_attention_matmul_bounds_zonotope` (attention `Q@K^T`) and
    ///    `try_ffn_swiglu_bounds_zonotope` (SwiGLU `MulBinary`) and merge the
    ///    result through [`super::dispatch::intersect_zonotope_ibp`], so those
    ///    paths can only be tighter than plain interval arithmetic. Those helpers
    ///    rebuild a fresh zonotope at the *shared base* of the Q and K
    ///    projections, which is where the Q/K correlation lives.
    /// 2. **Carrying symbols through the normalization — the one thing this pass
    ///    does that they do not — measures WORSE, not better.** See
    ///    `graph_alpha/zonotope/attention.rs:108-116`: on Qwen3-0.6B, propagating
    ///    the zonotope through LayerNorm gives `Q@K^T` width `2.44e3`, versus
    ///    `1e-6`..`1e-3` when a fresh zonotope is started at the LayerNorm output.
    ///    `layer_norm_affine`'s relaxation error swamps the correlation it keeps.
    ///    The tests in `propagate_zonotope_vs_production_ibp` below reproduce that
    ///    ordering on a hand-checkable fixture.
    /// 3. **It has no floor.** Unlike the node-local helpers, this pass publishes
    ///    its output zonotope verbatim — there is no `intersect_zonotope_ibp`
    ///    anywhere in it — so routing it could *widen* a bound rather than only
    ///    tighten it. Two further blockers before it could ever carry a verdict:
    ///    the Softmax arm's own comment below disclaims global soundness, and the
    ///    error-symbol count is uncapped (it scales with input size).
    ///
    /// If you are here looking for attention tightness, the live levers are the
    /// two node-local helpers above and the BilinearCrown McCormick alphas in
    /// `graph_alpha/propagate_dag` — not this function.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (center values)
    /// * `epsilon` - Perturbation epsilon for input
    ///
    /// # Returns
    /// Bounds as a BoundedTensor (converted from zonotope at output).
    ///
    /// # Supported Operations
    /// - Linear layers (preserves zonotope form exactly)
    /// - AddConstant, MulConstant (preserves zonotope form exactly)
    /// - MatMul for Q@K^T patterns (uses matmul_transposed)
    /// - Other ops: falls back to IBP and converts to zonotope (loses correlations)
    ///
    /// # Limitations
    /// - Best effort for tensors with >=2 dims (..., seq, dim)
    /// - Larger inputs may require significant memory for error symbols
    #[inline]
    pub fn propagate_zonotope(&self, input: &BoundedTensor, epsilon: f32) -> Result<BoundedTensor> {
        self.propagate_zonotope_with_options(input, epsilon, ZonotopePropagationOptions::default())
    }

    /// Propagate bounds through the graph using zonotopes with operator-local
    /// fallback controls.
    #[inline]
    pub fn propagate_zonotope_with_options(
        &self,
        input: &BoundedTensor,
        _epsilon: f32,
        options: ZonotopePropagationOptions,
    ) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            // Empty graph: nothing to do.
            return Ok(input.clone());
        }

        // Get execution order
        let exec_order = self.exec_order()?;

        // Zonotope propagation works best on 2D, but we support batched sequence tensors too.
        let input_shape = input.shape();
        if input_shape.len() < 2 {
            debug!(
                "GraphNetwork zonotope: input shape {:?} has <2 dims, falling back to IBP",
                input_shape
            );
            return self.propagate_ibp(input);
        }

        // Reject non-finite input early: if a previous block produced NaN/Inf,
        // zonotope construction is not meaningful. Return a fallback-class error
        // so the caller (e.g. Whisper verifier) can recover via IBP. Part of #3548.
        if input.has_overflow() {
            return Err(NyError::UnsupportedConfiguration(
                "zonotope propagation requires finite input bounds".to_string(),
            ));
        }

        // Create input zonotope with per-position error symbols derived from the current bounds.
        // This ensures zonotope propagation remains compatible with compositional bounds where
        // per-element radii may not equal the original epsilon.
        let input_zonotope = ZonotopeTensor::from_bounded_tensor_per_position(input)?;
        debug!(
            "GraphNetwork zonotope: created input zonotope with {} error terms, shape {:?}",
            input_zonotope.n_error_terms(),
            input_zonotope.shape()
        );

        // Store zonotopes for each node's output
        let mut zonotope_cache: std::collections::HashMap<String, ZonotopeTensor> =
            std::collections::HashMap::new();

        // Also store interval bounds for fallback
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        // Process nodes in topological order
        for node_name in exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            debug!(
                "GraphNetwork zonotope: processing node {} ({})",
                node_name,
                node.layer.layer_type()
            );

            // Try to propagate as zonotope; fall back to IBP if not supported
            let result =
                self.propagate_zonotope_node(node, &input_zonotope, &zonotope_cache, options);

            match result {
                Ok(z) => {
                    debug!(
                        "GraphNetwork zonotope: node {} output zonotope with {} error terms, max_width {}",
                        node_name,
                        z.n_error_terms(),
                        z.max_width()
                    );
                    bounds_cache.insert(node_name.clone(), z.to_bounded_tensor()?);
                    zonotope_cache.insert(node_name.clone(), z);
                }
                // #3166, #3548: UnsupportedOp/UnsupportedConfiguration/NumericalInstability:
                // zonotope propagation not available or produced non-finite values for
                // this node. IBP fallback is sound — it is the conservative degradation
                // for optional tightening paths. Other errors (SoundnessRefusal, etc.)
                // must propagate (#3106).
                Err(e @ NyError::UnsupportedOp(_))
                | Err(e @ NyError::UnsupportedConfiguration(_))
                | Err(e @ NyError::NumericalInstability(_)) => {
                    debug!(
                        "GraphNetwork zonotope: node {} falling back to IBP: {}",
                        node_name, e
                    );

                    // Use unified dispatch for IBP fallback (#2405).
                    // This correctly handles Concat before is_binary().
                    let ibp_bounds = dispatch_ibp_for_node(node, node_name, &mut |name| {
                        Ok(self.bounds_ref(name, input, &bounds_cache)?.clone())
                    })?;

                    // Convert IBP bounds to zonotope (loses correlation info).
                    // For attention `softmax @ V`, this is the Packet C seam:
                    // non-transposed MatMul falls back to interval bounds, then
                    // the interval is reintroduced as a single shared-error
                    // zonotope for the remaining affine suffix.
                    let z = ZonotopeTensor::from_bounded_tensor(&ibp_bounds);
                    if matches!(&node.layer, Layer::MatMul(matmul) if !matmul.transpose_b) {
                        debug!(
                            "GraphNetwork zonotope: node {} non-transposed MatMul fallback width {} -> re-zonotized width {} with {} error term(s)",
                            node_name,
                            ibp_bounds.max_width(),
                            z.max_width(),
                            z.n_error_terms()
                        );
                    }
                    bounds_cache.insert(node_name.clone(), ibp_bounds);
                    zonotope_cache.insert(node_name.clone(), z);
                    // Derived from IBP, not zonotope - it's derived from IBP
                }
                Err(e) => return Err(e),
            }

            // NaN firewall (#2812 Slice 4, #2706).
            if let Some(output_bounds) = bounds_cache.get(node_name) {
                check_nan_firewall(
                    output_bounds,
                    "Zonotope IBP",
                    node_name,
                    node.layer.layer_type(),
                )?;
            }
        }

        // Get output node
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        // Return bounds from output zonotope
        let output = zonotope_cache.get(output_node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;

        info!(
            "GraphNetwork zonotope: final output has {} error terms, max_width {}",
            output.n_error_terms(),
            output.max_width()
        );

        output.to_bounded_tensor()
    }

    /// Try to propagate a single node using zonotopes.
    ///
    /// Returns Ok(ZonotopeTensor) if the operation is supported, Err otherwise.
    pub(crate) fn propagate_zonotope_node(
        &self,
        node: &GraphNode,
        input_zonotope: &ZonotopeTensor,
        zonotope_cache: &std::collections::HashMap<String, ZonotopeTensor>,
        options: ZonotopePropagationOptions,
    ) -> Result<ZonotopeTensor> {
        // Helper to get zonotope for an input
        let get_zonotope = |name: &str| -> Result<ZonotopeTensor> {
            if name == NETWORK_INPUT {
                Ok(input_zonotope.clone())
            } else if let Some(z) = zonotope_cache.get(name) {
                Ok(z.clone())
            } else {
                Err(NyError::InvalidSpec(format!(
                    "No zonotope found for node {}",
                    name
                )))
            }
        };
        let get_unary_zonotope = || -> Result<ZonotopeTensor> {
            let input_name = node.require_unary_input()?;
            get_zonotope(input_name)
        };
        let get_binary_zonotopes = || -> Result<(ZonotopeTensor, ZonotopeTensor)> {
            let (input_a_name, input_b_name) = node.require_binary_inputs()?;
            Ok((get_zonotope(input_a_name)?, get_zonotope(input_b_name)?))
        };

        match &node.layer {
            // Linear layer: z = W @ z + b
            Layer::Linear(linear) => {
                let input_z = get_unary_zonotope()?;
                input_z.linear(linear.weight(), linear.bias())
            }

            // AddConstant: z + c
            Layer::AddConstant(add_const) => {
                let input_z = get_unary_zonotope()?;
                input_z.add_constant(&add_const.constant)
            }

            // MulConstant: z * c
            Layer::MulConstant(mul_const) => {
                let input_z = get_unary_zonotope()?;
                input_z.mul_constant(&mul_const.constant)
            }

            // MatMul: Q @ K^T or probs @ V in attention
            Layer::MatMul(matmul) => {
                let (q_z, k_z) = get_binary_zonotopes()?;

                if !matmul.transpose_b {
                    // `softmax @ V` commonly reaches this path after the Softmax
                    // node has fallen back to interval bounds and been reintroduced
                    // as a zonotope with fresh, disjoint error symbols. Use the
                    // conservative disjoint-symbol bilinear path instead of faking
                    // shared-symbol provenance (#318 Packet C). Keep the previous
                    // interval fallback if it is narrower on this node.
                    return propagate_disjoint_matmul(matmul, &q_z, &k_z);
                }

                // Check if Q and K share error symbols
                if q_z.n_error_terms() != k_z.n_error_terms() {
                    debug!(
                        "MatMul zonotope: Q has {} errors, K has {}, need to expand",
                        q_z.n_error_terms(),
                        k_z.n_error_terms()
                    );
                    let (q_expanded, k_expanded) = q_z.expand_to_match(&k_z)?;
                    return q_expanded.matmul_transposed(&k_expanded);
                }

                // Both have same error symbols - this is the ideal case for Q@K^T
                let out = q_z.matmul_transposed(&k_z)?;
                if let Some(scale) = matmul.scale {
                    let scale_tensor = ndarray::ArrayD::from_elem(out.shape().to_vec(), scale);
                    out.mul_constant(&scale_tensor)
                } else {
                    Ok(out)
                }
            }

            // Add: z1 + z2 (element-wise)
            Layer::Add(_) => {
                let (a_z, b_z) = get_binary_zonotopes()?;

                // Expand to match if needed
                let (a_expanded, b_expanded) = a_z.expand_to_match(&b_z)?;
                a_expanded.add(&b_expanded)
            }

            // Reshape: preserves correlations perfectly (just rearranges elements)
            Layer::Reshape(reshape_layer) => {
                let input_z = get_unary_zonotope()?;

                // Compute output shape from reshape layer
                let input_shape = input_z.shape();
                let output_shape = reshape_layer.compute_output_shape(input_shape)?;

                input_z.reshape(&output_shape)
            }

            // Tile: preserves correlations (duplicated elements share error symbols)
            // Essential for GQA where K/V heads are tiled to match Q heads
            Layer::Tile(tile_layer) => {
                let input_z = get_unary_zonotope()?;

                // Compute actual axis (handle negative indexing)
                let ndim = input_z.shape().len();
                let axis = crate::layers::common::resolve_axis_i32(
                    tile_layer.axis,
                    ndim,
                    "Tile (zonotope IBP)",
                )?;

                input_z.tile(axis, tile_layer.reps)
            }

            // Transpose: preserves correlations (just permutes axes)
            Layer::Transpose(transpose_layer) => {
                let input_z = get_unary_zonotope()?;

                // Check if it's a simple swap of last two dimensions
                let ndim = transpose_layer.axes.len();
                if ndim >= 2
                    && transpose_layer.axes[ndim - 2] == ndim - 1
                    && transpose_layer.axes[ndim - 1] == ndim - 2
                {
                    input_z.transpose_last_two()
                } else {
                    Err(NyError::UnsupportedOp(format!(
                        "Zonotope transpose only supports swapping last two dims, got axes {:?}",
                        transpose_layer.axes
                    )))
                }
            }

            // LayerNorm: use affine approximation to preserve correlations
            Layer::LayerNorm(ln) => {
                let input_z = get_unary_zonotope()?;
                match ln.mode {
                    LayerNormMode::Standard => input_z.layer_norm_affine(&ln.ny, &ln.beta, ln.eps),
                    LayerNormMode::MeanOnly => {
                        input_z.layer_norm_affine_mean_only(&ln.ny, &ln.beta)
                    }
                }
            }

            // SiLU: use affine approximation to preserve correlations
            Layer::SiLU(_) => {
                let input_z = get_unary_zonotope()?;
                input_z.silu_affine()
            }
            // GELU: use affine approximation to preserve correlations
            // Uses correct GELU math (not SiLU). Fix for #2470.
            Layer::GELU(g) => {
                let input_z = get_unary_zonotope()?;
                let use_tanh = matches!(
                    g.approximation,
                    crate::layers::softmax::GeluApproximation::Tanh
                );
                input_z.gelu_affine(use_tanh)
            }

            // MulBinary: element-wise multiplication (needed for SwiGLU)
            // z1 ⊙ z2 = silu(gate) * up, exploits shared error symbols
            Layer::MulBinary(_) => {
                let (z1, z2) = get_binary_zonotopes()?;

                z1.mul_elementwise(&z2)
            }

            // Softmax: use affine approximation to preserve correlations
            // This linearizes softmax around the center and adds a conservative error term;
            // the result is still heuristic and not a proof of global soundness.
            Layer::Softmax(s) => {
                if matches!(
                    options.softmax_mode(),
                    ZonotopeSoftmaxMode::IntervalFallback
                ) {
                    // #318 Packet B: cut the zonotope pipeline at Softmax so the
                    // existing node-local IBP fallback can isolate whether the
                    // widener is Softmax itself or a later attention operator.
                    return Err(NyError::UnsupportedOp(
                        "Zonotope Softmax explicitly configured to fall back to IBP".to_string(),
                    ));
                }
                let input_z = get_unary_zonotope()?;
                match input_z.softmax_affine(s.axis) {
                    Ok(z) => Ok(z),
                    Err(NyError::InvalidSpec(msg)) if msg.contains("not yet implemented") => {
                        Err(NyError::UnsupportedConfiguration(msg))
                    }
                    Err(e) => Err(e),
                }
            }

            // CausalSoftmax: use affine approximation (same as Softmax for now)
            Layer::CausalSoftmax(cs) => {
                if matches!(
                    options.softmax_mode(),
                    ZonotopeSoftmaxMode::IntervalFallback
                ) {
                    return Err(NyError::UnsupportedOp(
                        "Zonotope CausalSoftmax explicitly configured to fall back to IBP"
                            .to_string(),
                    ));
                }
                let input_z = get_unary_zonotope()?;
                match input_z.softmax_affine_causal(cs.axis) {
                    Ok(z) => Ok(z),
                    Err(NyError::InvalidSpec(msg)) if msg.contains("not yet implemented") => {
                        Err(NyError::UnsupportedConfiguration(msg))
                    }
                    Err(e) => Err(e),
                }
            }

            // Operations that don't preserve zonotope form well
            Layer::ReLU(_)
            | Layer::Tanh(_)
            | Layer::Sigmoid(_)
            | Layer::Tan(_)
            | Layer::Arctan(_)
            | Layer::Sqrt(_)
            | Layer::PowConstant(_) => {
                // These operations break zonotope form — fall back to IBP (#3117).
                Err(NyError::UnsupportedOp(format!(
                    "Operation {} not supported for zonotope propagation",
                    node.layer.layer_type()
                )))
            }

            // Other operations — not yet implemented (#3117).
            _ => Err(NyError::UnsupportedOp(format!(
                "Operation {} not yet implemented for zonotope propagation",
                node.layer.layer_type()
            ))),
        }
    }
}

/// Evidence for the "route it or delete it" question on [`GraphNetwork::propagate_zonotope`]
/// (#orphans-lies-1).
///
/// The claim under test is the one in this module's own doc: that a whole-graph
/// zonotope forward gives "tighter bounds for Q@K^T in attention than IBP", and
/// therefore that its zero production callers are a missed tightness win.
///
/// On an attention-shaped fixture it does not hold. `propagate_ibp` is **not**
/// correlation-blind: it already runs the node-local zonotope tightenings
/// (`try_attention_matmul_bounds_zonotope` for `MatMul{transpose_b}`,
/// `try_ffn_swiglu_bounds_zonotope` for `MulBinary`) and merges them through
/// [`super::dispatch::intersect_zonotope_ibp`], so it can only be tighter than
/// plain interval arithmetic. Those helpers rebuild a FRESH zonotope at the
/// shared base of the Q and K projections, which is where the Q/K correlation
/// actually lives — while this whole-graph pass insists on carrying symbols
/// *through* the normalization via `layer_norm_affine`, whose relaxation error
/// swamps the correlation it preserves. That is the measurement already recorded
/// at `graph_alpha/zonotope/attention.rs:108-116` (Qwen3-0.6B: Q@K^T width
/// 2.44e3 propagating through LayerNorm vs 1e-6..1e-3 restarting after it), and
/// these tests reproduce the same ordering on a fixture small enough to check by
/// hand against concrete forward evaluations.
///
/// MEASURED here (`_input -> LayerNorm -> {q, k} -> q @ k^T`, `max_width`):
///
/// | input radius | `propagate_ibp` | `propagate_zonotope` | ratio |
/// |---|---|---|---|
/// | 0.001 | 8.744907e-2 | 1.365502e-1 | 1.56x WORSE |
/// | 0.02  | 2.230007e0  | 5.794865e0  | 2.60x WORSE |
/// | 0.1   | 2.110639e1  | 3.037820e2  | 14.4x WORSE |
///
/// And the control, at radius 0.02: the same graph with Q and K fed by two
/// SEPARATE LayerNorms (no shared base, so nothing for the node-local zonotope
/// to exploit) gives `4.083443e0` against the shared-base `2.230007e0` — i.e.
/// `propagate_ibp` really is 1.83x tighter *because* of the correlation
/// tightening, so the comparison above is against the correlation-aware path.
#[cfg(test)]
mod propagate_zonotope_vs_production_ibp {
    use crate::layers::{Layer, LayerNormLayer, LinearLayer, MatMulLayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use ndarray::{Array2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    const SEQ: usize = 4;
    const DIM: usize = 6;
    const EPS: f32 = 1e-5;

    fn projection(seed: f32) -> Array2<f32> {
        Array2::from_shape_fn((DIM, DIM), |(o, i)| {
            let t = seed + o as f32 * 0.37 - i as f32 * 0.19;
            0.5 * t.sin() + 0.25 * (t * 1.7).cos()
        })
    }

    /// `_input -> ln -> {q, k} -> scores = q @ k^T`.
    ///
    /// This is the shape ny-build emits for a traced attention block: one shared
    /// normalized activation feeding two projections whose product is the score
    /// matrix. It is exactly the pattern the module doc claims the whole-graph
    /// zonotope wins on.
    fn attention_scores_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "ln",
            Layer::LayerNorm(LayerNormLayer::new_default(DIM, EPS).expect("layernorm")),
        ));
        graph.add_node(GraphNode::new(
            "q",
            Layer::Linear(LinearLayer::new(projection(0.3), None).expect("q proj")),
            vec!["ln".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "k",
            Layer::Linear(LinearLayer::new(projection(1.9), None).expect("k proj")),
            vec!["ln".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "scores",
            Layer::MatMul(MatMulLayer::new(true, None)),
            "q",
            "k",
        ));
        graph.set_output("scores");
        graph
    }

    fn input_box(radius: f32) -> BoundedTensor {
        let center = ArrayD::from_shape_fn(IxDyn(&[SEQ, DIM]), |ix| {
            let (s, d) = (ix[0] as f32, ix[1] as f32);
            0.8 * (s * 1.3 + d * 0.7).sin() + 0.2 * d
        });
        BoundedTensor::new(center.mapv(|v| v - radius), center.mapv(|v| v + radius))
            .expect("input box")
    }

    /// Concrete forward, written out independently of ny's propagation code so
    /// the enclosure check is not circular: LayerNorm over the last axis, two
    /// `x @ W^T` projections, then `q @ k^T`.
    fn concrete_scores(x: &Array2<f32>) -> Array2<f32> {
        let mut normalized = Array2::<f32>::zeros((SEQ, DIM));
        for s in 0..SEQ {
            let row = x.row(s);
            let mean = row.sum() / DIM as f32;
            let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / DIM as f32;
            let inv = 1.0 / (var + EPS).sqrt();
            for d in 0..DIM {
                normalized[[s, d]] = (row[d] - mean) * inv;
            }
        }
        let q = normalized.dot(&projection(0.3).t());
        let k = normalized.dot(&projection(1.9).t());
        q.dot(&k.t())
    }

    /// Deterministic sample points spanning the input box: both corners, the
    /// center, and interior points that vary the sign pattern per coordinate.
    fn samples(input: &BoundedTensor) -> Vec<Array2<f32>> {
        let lower = input.lower();
        let upper = input.upper();
        let at = |f: &dyn Fn(usize, usize) -> f32| {
            Array2::from_shape_fn((SEQ, DIM), |(s, d)| {
                let (lo, hi) = (lower[[s, d]], upper[[s, d]]);
                lo + f(s, d).clamp(0.0, 1.0) * (hi - lo)
            })
        };
        vec![
            at(&|_, _| 0.0),
            at(&|_, _| 1.0),
            at(&|_, _| 0.5),
            at(&|s, d| ((s + d) % 2) as f32),
            at(&|s, d| ((s * DIM + d) % 3) as f32 / 2.0),
            at(&|s, d| (((s * 7 + d * 5) % 11) as f32) / 10.0),
        ]
    }

    fn assert_encloses(bounds: &BoundedTensor, input: &BoundedTensor, label: &str) {
        for (idx, x) in samples(input).iter().enumerate() {
            let out = concrete_scores(x);
            for i in 0..SEQ {
                for j in 0..SEQ {
                    let v = out[[i, j]];
                    assert!(
                        v >= bounds.lower()[[i, j]] - 1e-4 && v <= bounds.upper()[[i, j]] + 1e-4,
                        "{label} sample {idx}: scores[{i},{j}]={v} escapes [{}, {}]",
                        bounds.lower()[[i, j]],
                        bounds.upper()[[i, j]],
                    );
                }
            }
        }
    }

    /// Both paths are enclosures — the orphan is not unsound, it is redundant.
    #[ntest::timeout(30000)]
    #[test]
    fn both_paths_enclose_concrete_attention_scores() {
        let graph = attention_scores_graph();
        for radius in [0.001_f32, 0.02, 0.1] {
            let input = input_box(radius);
            let ibp = graph.propagate_ibp(&input).expect("production IBP");
            assert_encloses(&ibp, &input, "propagate_ibp");
            let zono = graph
                .propagate_zonotope(&input, radius)
                .expect("whole-graph zonotope");
            assert_encloses(&zono, &input, "propagate_zonotope");
        }
    }

    /// The tightness claim fails on the shape it was written for: the whole-graph
    /// zonotope is never tighter than production IBP here, and is much looser as
    /// the box grows.
    ///
    /// This is the load-bearing assertion behind the recommendation to DELETE
    /// rather than route `propagate_zonotope`. If a future change makes the
    /// whole-graph pass genuinely win on this fixture, this test fails and the
    /// recommendation should be revisited — that is the intent.
    #[ntest::timeout(30000)]
    #[test]
    fn whole_graph_zonotope_is_not_tighter_than_production_ibp_on_attention() {
        let graph = attention_scores_graph();
        for radius in [0.001_f32, 0.02, 0.1] {
            let input = input_box(radius);
            let ibp = graph.propagate_ibp(&input).expect("production IBP");
            let zono = graph
                .propagate_zonotope(&input, radius)
                .expect("whole-graph zonotope");
            println!(
                "radius={radius:.4}: propagate_ibp max_width={:.6e}  propagate_zonotope max_width={:.6e}",
                ibp.max_width(),
                zono.max_width()
            );
            assert!(
                zono.max_width() >= ibp.max_width() * (1.0 - 1e-3),
                "radius {radius}: whole-graph zonotope ({:.6e}) beat production IBP ({:.6e}) — \
                 the orphan may be worth routing after all; see the module note",
                zono.max_width(),
                ibp.max_width(),
            );
        }
    }

    /// Production IBP's correlation awareness is not incidental: removing the
    /// shared-base link (Q and K fed by two *different* normalizations) is what
    /// a correlation-blind path would look like, and it is measurably worse.
    /// This pins that `propagate_ibp` really is exercising the node-local
    /// zonotope tightening on the shared-base fixture, so the comparison above
    /// is against the correlation-aware path and not a strawman.
    #[ntest::timeout(30000)]
    #[test]
    fn production_ibp_exercises_the_node_local_zonotope_on_the_shared_base() {
        let shared = attention_scores_graph();

        let mut split = GraphNetwork::new();
        split.add_node(GraphNode::from_input(
            "ln_q",
            Layer::LayerNorm(LayerNormLayer::new_default(DIM, EPS).expect("layernorm")),
        ));
        split.add_node(GraphNode::from_input(
            "ln_k",
            Layer::LayerNorm(LayerNormLayer::new_default(DIM, EPS).expect("layernorm")),
        ));
        split.add_node(GraphNode::new(
            "q",
            Layer::Linear(LinearLayer::new(projection(0.3), None).expect("q proj")),
            vec!["ln_q".to_string()],
        ));
        split.add_node(GraphNode::new(
            "k",
            Layer::Linear(LinearLayer::new(projection(1.9), None).expect("k proj")),
            vec!["ln_k".to_string()],
        ));
        split.add_node(GraphNode::binary(
            "scores",
            Layer::MatMul(MatMulLayer::new(true, None)),
            "q",
            "k",
        ));
        split.set_output("scores");

        let input = input_box(0.02);
        let shared_width = shared
            .propagate_ibp(&input)
            .expect("shared-base IBP")
            .max_width();
        let split_width = split
            .propagate_ibp(&input)
            .expect("split-base IBP")
            .max_width();
        println!(
            "shared-base IBP max_width={shared_width:.6e}  split-base IBP max_width={split_width:.6e}"
        );
        assert!(
            shared_width <= split_width,
            "shared-base IBP ({shared_width:.6e}) should be at least as tight as split-base \
             ({split_width:.6e}); if not, the node-local zonotope tightening did not fire and \
             the orphan comparison is against a strawman"
        );
    }
}
