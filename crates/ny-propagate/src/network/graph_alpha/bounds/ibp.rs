// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP (interval bound propagation) forward pass for `GraphNetwork`.
//!
//! Extracted from `bounds/mod.rs` to keep files under 500 lines (#3396).

use crate::layers::{BoundPropagation, Layer};
use crate::network::core::graph::ibp::dispatch::{
    check_nan_firewall, classify_node_inputs, ResolvedInputNames,
};

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;

use crate::network::core::GraphNetwork;

/// Dispatch a single layer's IBP through the engine-aware path when the layer
/// is a GEMM-heavy type (Linear, Conv1d, Conv2d, ConvTranspose1d,
/// ConvTranspose2d). All other layers fall back to the CPU-only trait method.
/// Mirrors the sequential `propagate_layer_ibp_with_engine` in
/// `network/ibp/forward.rs`.
pub(super) fn propagate_node_ibp_with_engine(
    layer: &Layer,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    match layer {
        // Linear/MatMul already round their IBP endpoints OUTWARD internally (1-D:
        // f64 accumulation + directed cast; N-D/GEMM: round_for_soundness_n_ulps), so
        // their node boxes already enclose the true pre-activation.
        Layer::Linear(l) => l.propagate_ibp_with_engine(input, engine),
        // Conv2d's PLAIN propagate_ibp is a round-to-NEAREST f32 GEMM with no directed
        // rounding and no Higham term — it can produce a node box that EXCLUDES the
        // true pre-activation under cancellation (demonstrated: W=[2^24,-1,-2^24] →
        // node [0,0] but true −1). Feeding that to the ReLU `l>=0 → identity` guard
        // mis-classifies a truly-unstable neuron as stable-active → a FALSE VERIFIED.
        // Node bounds MUST be the SOUND (abssum-Higham, directed) forward.
        // (#vnncomp-aw-soundness self-audit — intermediate-bound false-proof.)
        Layer::Conv2d(c) => c.propagate_ibp_sound_with_engine(input, engine),
        // Conv1d / ConvTranspose1d / ConvTranspose2d share the identical round-to-nearest
        // f32 node-bound gap; each now has the same Higham-sound forward.
        Layer::Conv1d(c) => c.propagate_ibp_sound_with_engine(input, engine),
        Layer::ConvTranspose1d(c) => c.propagate_ibp_sound_with_engine(input, engine),
        Layer::ConvTranspose2d(c) => c.propagate_ibp_sound_with_engine(input, engine),
        _ => layer.propagate_ibp(input),
    }
}

impl GraphNetwork {
    /// Collect IBP bounds at each node in the graph.
    pub fn collect_node_bounds(
        &self,
        input: &BoundedTensor,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_node_bounds_core(input, false, None, None, false)
    }

    /// Collect per-node activations for a POINT (degenerate) input, collapsing
    /// every node's output to its interval center before caching so a point input
    /// stays a point through every node (mirrors the faithful `propagate_concrete_point`
    /// center-collapse, #cgan-eval). Returns the full per-node cache whose
    /// `.center()`/`.lower()` equal the true network activation of that node to ~ULP.
    ///
    /// Diagnostic-only (used by the `NY_LOOSENESS_PROBE`): non-soundness-critical,
    /// never feeds a verdict. The caller must pass a degenerate box (lower == upper).
    pub fn collect_node_activations_pointwise(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_node_bounds_core(input, false, engine, None, true)
    }

    /// Collect IBP bounds at each node, with optional GPU engine acceleration.
    ///
    /// When an engine is provided, Linear, Conv1d, Conv2d, ConvTranspose1d,
    /// and ConvTranspose2d layers dispatch through their engine-aware IBP paths
    /// (potentially using GPU GEMM). All other layers fall back to CPU. (#4174)
    pub fn collect_node_bounds_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_node_bounds_core(input, false, engine, None, false)
    }

    /// Collect IBP bounds at each node, aborting when the deadline is exceeded.
    pub fn collect_node_bounds_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_node_bounds_core(input, false, engine, deadline, false)
    }

    /// Collect IBP bounds at each node, allowing sqrt inputs to be clamped.
    ///
    /// This is intended for soundness scans that need to detect negative-domain
    /// sqrt nodes without failing the entire pass.
    pub(crate) fn collect_node_bounds_allowing_negative_sqrt(
        &self,
        input: &BoundedTensor,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_node_bounds_core(input, true, None, None, false)
    }

    fn collect_node_bounds_core(
        &self,
        input: &BoundedTensor,
        allow_negative_sqrt: bool,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        collapse_to_center: bool,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        // NOTE on the L2/Cauchy–Schwarz lever inside CROWN: an outer CROWN scope
        // has DISABLED the lever (see `crate::l2_lever_gate`), and this node-bound
        // collection intentionally inherits that. Enabling the (now-cheap) lever
        // here was measured to add cost on deep block-wise CROWN (table_transformer)
        // while producing ZERO additional tightening on the frontier CROWN tests:
        // their final bound is already floored with a top-level plain IBP pass
        // (`nn-verify::floor_with_ibp`), which carries the lever, so the relaxation
        // never needs the sphere on the intermediate boxes. Keeping it disabled here
        // is sound (the box already encloses the true value) and avoids re-paying any
        // per-collection cost. See the L2 lever investigation notes.
        let exec_order = self.exec_order()?;
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        for node_name in exec_order {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(NyError::DeadlineExceeded(format!(
                    "Graph IBP: deadline exceeded before node '{}'",
                    node_name
                )));
            }
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            let output_bounds = self.node_ibp_step(
                node_name,
                node,
                input,
                &bounds_cache,
                allow_negative_sqrt,
                engine,
            )?;

            // NaN firewall (#3768): guard the 5th IBP path like the other 4.
            check_nan_firewall(
                &output_bounds,
                "collect_node_bounds",
                node_name,
                node.layer.layer_type(),
            )?;

            // Faithful point forward (#cgan-eval): collapse to the interval center so a
            // point input stays degenerate through every node and per-node soundness
            // widening cannot be amplified downstream. Diagnostic use only.
            let output_bounds = if collapse_to_center {
                BoundedTensor::concrete(output_bounds.center())?
            } else {
                output_bounds
            };

            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        Ok(bounds_cache)
    }

    /// One node's IBP forward step, reading the node's input enclosures from
    /// `bounds_cache` (the network input box for `NETWORK_INPUT`). Extracted
    /// verbatim from [`Self::collect_node_bounds_core`] so the #stabilize
    /// downstream resweep can reuse the exact same sound per-node dispatch.
    fn node_ibp_step(
        &self,
        node_name: &str,
        node: &crate::network::GraphNode,
        input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
        allow_negative_sqrt: bool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        // Classify node arity via the shared accessor-based classifier (#2633).
        // This replaces 15 direct `node.inputs[N]` indexing sites with safe
        // GraphNode accessors. Layer-specific dispatch (zonotope tightening,
        // sqrt lenient mode) is handled within each arity arm below.
        let classified = classify_node_inputs(node, node_name)?;

        let output_bounds = match classified {
            ResolvedInputNames::Unary(name) => {
                let bounds = self.bounds_ref(name, input, bounds_cache)?;
                match &node.layer {
                    Layer::Where(w) if w.has_embedded_constants() => {
                        w.propagate_ibp_with_condition(bounds)?
                    }
                    Layer::Sqrt(sqrt) if allow_negative_sqrt => {
                        sqrt.propagate_ibp_lenient(bounds)?
                    }
                    // Route GEMM-heavy layers through engine-aware IBP (#4174).
                    _ => propagate_node_ibp_with_engine(&node.layer, bounds, engine)?,
                }
            }
            ResolvedInputNames::Binary(name_a, name_b) => {
                let input_a = self.bounds_ref(name_a, input, bounds_cache)?;
                let input_b = self.bounds_ref(name_b, input, bounds_cache)?;
                match &node.layer {
                    Layer::MatMul(matmul) if matmul.transpose_b => {
                        if let Some(tighter) =
                            self.try_attention_matmul_bounds_zonotope(node, input, bounds_cache)?
                        {
                            tighter
                        } else {
                            node.layer.propagate_ibp_binary(input_a, input_b)?
                        }
                    }
                    Layer::MulBinary(_) => {
                        // Try zonotope tightening for SwiGLU pattern (up * silu(gate)).
                        // Intersect with plain IBP: both sound, keep the tighter
                        // per element (no regression where the zonotope is looser).
                        let ibp = node.layer.propagate_ibp_binary(input_a, input_b)?;
                        match self.try_ffn_swiglu_bounds_zonotope(node, input, bounds_cache)? {
                            Some(zono) => {
                                crate::network::core::graph::ibp::dispatch::intersect_zonotope_ibp(
                                    zono, ibp,
                                )
                            }
                            None => ibp,
                        }
                    }
                    _ => node.layer.propagate_ibp_binary(input_a, input_b)?,
                }
            }
            ResolvedInputNames::Ternary(name_a, name_b, name_c) => {
                let input_a = self.bounds_ref(name_a, input, bounds_cache)?;
                let input_b = self.bounds_ref(name_b, input, bounds_cache)?;
                let input_c = self.bounds_ref(name_c, input, bounds_cache)?;
                node.layer
                    .propagate_ibp_ternary(input_a, input_b, input_c)?
            }
            ResolvedInputNames::NaryConcat {
                dynamic_inputs,
                has_constants,
            } => {
                let concat = match &node.layer {
                    Layer::Concat(c) => c,
                    _ => {
                        return Err(NyError::InternalError(format!(
                            "classify_node_inputs returned NaryConcat for non-Concat node '{}'",
                            node_name
                        )));
                    }
                };
                let owned_inputs: Vec<BoundedTensor> = if has_constants {
                    let ci = concat.constant_inputs.as_ref().ok_or_else(|| {
                        NyError::InternalError(format!(
                            "classify_node_inputs signaled has_constants but Concat '{}' \
                                 has no constant_inputs",
                            node_name
                        ))
                    })?;
                    let mut graph_idx = 0;
                    ci.iter()
                        .map(|const_opt| {
                            if let Some(constant) = const_opt {
                                Ok(constant.clone())
                            } else {
                                let name = dynamic_inputs.get(graph_idx).ok_or_else(|| {
                                    NyError::InternalError(format!(
                                        "Concat '{}': ran out of graph inputs at graph_idx {}",
                                        node_name, graph_idx
                                    ))
                                })?;
                                graph_idx += 1;
                                Ok(self.bounds_ref(name, input, bounds_cache)?.clone())
                            }
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    dynamic_inputs
                        .iter()
                        .map(|inp_name| Ok(self.bounds_ref(inp_name, input, bounds_cache)?.clone()))
                        .collect::<Result<Vec<_>>>()?
                };
                let input_refs: Vec<&BoundedTensor> = owned_inputs.iter().collect();
                concat.propagate_ibp_nary(&input_refs)?
            }
        };

        Ok(output_bounds)
    }

    /// #stabilize downstream recompute (dark `NY_STABILIZE=<secs>`): one forward
    /// IBP sweep in exec order that reads every node's INPUT enclosures from the
    /// caller's STORED bounds map and per-element INTERSECTS the one-step IBP
    /// image into that node's stored entry (shrink-only, union fallback on the
    /// impossible-disjoint case). This propagates a tightened (e.g. fixed-stable)
    /// pre-activation into every downstream stored entry so the next stabilize
    /// round's scan sees newly-stabilizable neurons.
    ///
    /// SOUND: each stored entry is a sound enclosure of its node's reachable set
    /// over the root box; the layer's sound (outward-rounded) IBP image of sound
    /// input enclosures also encloses that reachable set; the per-element
    /// intersection of two sound enclosures of the same set still encloses it.
    /// Any per-node failure (unsupported op, missing entry, shape mismatch, NaN
    /// ⇒ `intersection_per_element` returns `None`) keeps the stored sound bound
    /// untouched. Deadline expiry leaves a prefix of nodes tightened — every
    /// completed merge is sound.
    ///
    /// Returns the number of node entries merged.
    pub(crate) fn resweep_stored_bounds_ibp(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        bounds: &mut std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<usize> {
        let exec_order = self.exec_order()?;
        let mut merged = 0usize;
        for node_name in exec_order {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break; // partial work: every completed merge above is sound
            }
            let Some(node) = self.nodes.get(node_name) else {
                continue;
            };
            // Only tighten nodes that already have a stored enclosure; the map
            // is never extended (a new key would bypass the inherited contract).
            if !bounds.contains_key(node_name) {
                continue;
            }
            let step = match self.node_ibp_step(node_name, node, input, bounds, false, engine) {
                Ok(b) => b,
                // Keep the stored sound bound on any per-node failure.
                Err(_) => continue,
            };
            let Some(stored) = bounds.get(node_name) else {
                continue;
            };
            if stored.shape() != step.shape() {
                continue;
            }
            // Shrink-only merge (NaN ⇒ None ⇒ keep stored; disjoint ⇒ union).
            if let Some((tightened, _disjoint)) = stored.intersection_per_element(&step) {
                bounds.insert(node_name.clone(), tightened);
                merged += 1;
            }
        }
        Ok(merged)
    }
}
