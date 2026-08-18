// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP (interval bound propagation) forward pass for `GraphNetwork`.
//!
//! Extracted from `bounds/mod.rs` to keep files under 500 lines (#3396).

use crate::layers::{BoundPropagation, Layer};
use crate::network::core::graph::ibp::dispatch::{
    check_nan_firewall, check_nan_firewall_with_poll, classify_node_inputs, intersect_zonotope_ibp,
    intersect_zonotope_ibp_with_poll, ResolvedInputNames,
};

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;

use crate::network::core::GraphNetwork;

#[inline]
fn check_graph_alpha_ibp_deadline(deadline: Instant, node_name: &str, stage: &str) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(format!(
            "Graph IBP: deadline exceeded {stage} for node '{node_name}'"
        )));
    }
    Ok(())
}

fn concrete_center_with_deadline(
    bounds: &BoundedTensor,
    deadline: Option<Instant>,
    node_name: &str,
) -> Result<BoundedTensor> {
    if let Some(deadline) = deadline {
        let center = bounds.center_with_poll(|| {
            check_graph_alpha_ibp_deadline(deadline, node_name, "while centering output bounds")
        })?;
        BoundedTensor::concrete_with_poll(center, || {
            check_graph_alpha_ibp_deadline(deadline, node_name, "while centering output bounds")
        })
    } else {
        BoundedTensor::concrete(bounds.center())
    }
}

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
    propagate_node_ibp_with_engine_and_deadline(layer, input, engine, None)
}

/// Deadline-bearing graph-node IBP dispatch.
///
/// With a finite deadline, Conv1d, Conv2d, and ConvTranspose1d route their
/// certified pass through pollable CPU work and never enter the caller engine
/// or faer. The 1D certificate is a directed-f64 interval contraction; Conv2d's
/// finite-deadline certificate is the f64 dual-accumulator kernel (strictly
/// tighter than its deadline=None coefficient-abssum widening,
/// #cgan-conv-ibp-magnitude-floor).
pub(super) fn propagate_node_ibp_with_engine_and_deadline(
    layer: &Layer,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
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
        Layer::Conv2d(c) => c.propagate_ibp_sound_with_engine_and_deadline(input, engine, deadline),
        // Conv1d / ConvTranspose1d / ConvTranspose2d share the identical
        // round-to-nearest f32 node-bound gap. The 1D variants also carry the
        // graph authority through their pollable primary and directed-f64
        // certificate passes.
        Layer::Conv1d(c) => c.propagate_ibp_sound_with_engine_and_deadline(input, engine, deadline),
        Layer::ConvTranspose1d(c) => {
            c.propagate_ibp_sound_with_engine_and_deadline(input, engine, deadline)
        }
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
        self.collect_node_bounds_core(input, false, None, None, None, false)
    }

    /// Collect per-node activations for a POINT (degenerate) input, collapsing
    /// every node's output to its interval center before caching so a point input
    /// stays a point through every node (mirrors the faithful `propagate_concrete_point`
    /// center-collapse, #cgan-eval). Returns the full per-node cache whose
    /// `.center()`/`.lower()` equal the true network activation of that node to ~ULP.
    ///
    /// Current consumers are the `NY_LOOSENESS_PROBE` and the default-dark
    /// envelope-gradient steering heuristic. This cache is not a certified
    /// enclosure for any non-degenerate domain and must not be published as a
    /// verdict-bearing bound. The steering consumer can influence a later
    /// certified bound indirectly through its choice of alpha, which remains in
    /// the valid [0,1] relaxation domain. The caller must pass a degenerate box
    /// (lower == upper).
    pub fn collect_node_activations_pointwise(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_node_bounds_core(input, false, engine, None, None, true)
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
        self.collect_node_bounds_core(input, false, engine, None, None, false)
    }

    /// Collect IBP bounds at each node, aborting when the deadline is exceeded.
    pub fn collect_node_bounds_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_node_bounds_core(input, false, engine, deadline, deadline, false)
    }

    /// Collect IBP bounds at each node, allowing sqrt inputs to be clamped.
    ///
    /// This is intended for soundness scans that need to detect negative-domain
    /// sqrt nodes without failing the entire pass.
    pub(crate) fn collect_node_bounds_allowing_negative_sqrt(
        &self,
        input: &BoundedTensor,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_node_bounds_core(input, true, None, None, None, false)
    }

    /// `deadline` is the LOOP authority: it aborts the collection between nodes.
    /// `layer_deadline` is what reaches the layer kernels, and it is a different
    /// question. A finite layer deadline forces every Conv2d off im2col+GEMM onto
    /// `conv2d_ibp_forward_grouped_with_deadline`, a serial per-MAC pollable scalar
    /// contraction (`conv2d/ops_ibp_fwd.rs`), and makes `conv2d/bound.rs`'s
    /// `propagate_ibp_with_engine_and_deadline`
    /// discard the engine outright. The two routes are documented as
    /// "mathematically identical"; the finite-deadline one buys INTRA-node
    /// cancellation and pays for it in throughput.
    ///
    /// Verdict paths want that trade and keep passing `layer_deadline == deadline`.
    /// ATTACK-ONLY node collection: layer kernels keep their im2col/GEMM route.
    ///
    /// `attack_point_gradient` was measured at **8.5 s/step** on
    /// CIFAR100_resnet_large, which gives the upfront falsification lane ZERO
    /// gradient steps inside its 4 s slice. This module's own
    /// `point_vjp_batched_resnet.rs:9-12` records the same call on the same model
    /// family at **~93 ms/step** — a ~91x regression, and the cause is entirely
    /// the finite deadline reaching the kernels: 20 convs x 2 passes is ~718M
    /// serial per-MAC scalar iterations with `IxDyn` indexing, checked arithmetic
    /// and a poll counter, on one core.
    ///
    /// THE BOUND ONLY EVER TIGHTENS. The plain forward
    /// `conv2d_ibp_forward_grouped_with_deadline` is documented as
    /// "mathematically identical to `conv2d_ibp_forward_grouped`", and since
    /// #cgan-conv-ibp-magnitude-floor the finite-deadline SOUND arm is the
    /// certified f64 dual-accumulator kernel, whose enclosure is at least as
    /// tight as the deadline=None abssum-Higham construction — so this
    /// attack-only route (which skips the layer deadline) sees node boxes that
    /// are the same or LOOSER than the verdict route's, never tighter. What is
    /// given up is INTRA-node cancellation, so a timeout can overshoot by at
    /// most one node's kernel — the largest here is a 37.7M-MAC f32 im2col
    /// GEMM, single-digit milliseconds. The per-NODE check in the collection
    /// loop still fires.
    ///
    /// NEVER call this from a verdict path. It is admissible here only because
    /// `attack_point_gradient` carries NO verdict authority: it steers a search,
    /// and every candidate it produces still passes the trusted-ORT and true-f64
    /// admission gates before it can become a `sat`. A wrong or late attack
    /// gradient can only waste steps; it cannot manufacture a verdict.
    pub fn collect_node_bounds_attack_point(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_node_bounds_core(input, false, engine, deadline, None, false)
    }

    fn collect_node_bounds_core(
        &self,
        input: &BoundedTensor,
        allow_negative_sqrt: bool,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        layer_deadline: Option<Instant>,
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
                layer_deadline,
            )?;

            // NaN firewall (#3768): guard the 5th IBP path like the other 4.
            if let Some(deadline) = deadline {
                check_nan_firewall_with_poll(
                    &output_bounds,
                    "collect_node_bounds",
                    node_name,
                    node.layer.layer_type(),
                    || {
                        check_graph_alpha_ibp_deadline(
                            deadline,
                            node_name,
                            "while checking the NaN firewall",
                        )
                    },
                )?;
            } else {
                check_nan_firewall(
                    &output_bounds,
                    "collect_node_bounds",
                    node_name,
                    node.layer.layer_type(),
                )?;
            }

            // Faithful point forward (#cgan-eval): collapse to the interval center so a
            // point input stays degenerate through every node and per-node soundness
            // widening cannot be amplified downstream. Diagnostic use only.
            let output_bounds = if collapse_to_center {
                concrete_center_with_deadline(&output_bounds, deadline, node_name)?
            } else {
                output_bounds
            };

            if let Some(deadline) = deadline {
                check_graph_alpha_ibp_deadline(
                    deadline,
                    node_name,
                    "before caching output bounds",
                )?;
            }
            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        Ok(bounds_cache)
    }

    /// One node's IBP forward step, reading the node's input enclosures from
    /// `bounds_cache` (the network input box for `NETWORK_INPUT`). Extracted
    /// verbatim from [`Self::collect_node_bounds_core`] so the #stabilize
    /// downstream resweep can reuse the exact same sound per-node dispatch.
    pub(super) fn node_ibp_step(
        &self,
        node_name: &str,
        node: &crate::network::GraphNode,
        input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
        allow_negative_sqrt: bool,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
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
                    _ => propagate_node_ibp_with_engine_and_deadline(
                        &node.layer,
                        bounds,
                        engine,
                        deadline,
                    )?,
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
                                if let Some(deadline) = deadline {
                                    intersect_zonotope_ibp_with_poll(zono, ibp, || {
                                        check_graph_alpha_ibp_deadline(
                                            deadline,
                                            node_name,
                                            "while intersecting zonotope bounds",
                                        )
                                    })?
                                } else {
                                    intersect_zonotope_ibp(zono, ibp)
                                }
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
            let step =
                match self.node_ibp_step(node_name, node, input, bounds, false, engine, deadline) {
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
            // Under a finite deadline, build the candidate off-map and poll
            // bounded chunks. Expiry leaves this node uncommitted and preserves
            // the documented sound prefix of prior merges.
            let tightened = if let Some(deadline) = deadline {
                match stored.intersection_per_element_with_poll(&step, || {
                    check_graph_alpha_ibp_deadline(
                        deadline,
                        node_name,
                        "while intersecting stored bounds",
                    )
                }) {
                    Ok(result) => result,
                    Err(error) if error.is_deadline_exceeded() => break,
                    Err(error) => return Err(error),
                }
            } else {
                stored.intersection_per_element(&step)
            };
            if let Some((tightened, _disjoint)) = tightened {
                bounds.insert(node_name.clone(), tightened);
                merged += 1;
            }
        }
        Ok(merged)
    }
}
