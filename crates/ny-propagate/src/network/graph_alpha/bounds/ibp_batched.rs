// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-batched-interm — batched (multi-domain) intermediate-bounds forward.
//!
//! Design-doc slice S2 (`docs/LSNC_BATCH_TENSOR_DESIGN.md`): on the lsnc
//! input-split lane every rebound runs `collect_intermediate_bounds` once PER
//! DOMAIN under a rayon `par_iter` fan-out — per-domain `exec_order` walks,
//! per-node `String`-keyed `HashMap` lookups/inserts, and per-domain dispatch
//! classification, in per-domain work items of tens of µs. This module
//! computes the SAME per-node bounds for the WHOLE domain batch in one pass:
//!
//! * the graph structure work (exec order, input-arity classification, name →
//!   exec-index resolution, layer-arm selection) is resolved ONCE per batch
//!   instead of once per domain;
//! * per-domain per-node bounds are held in an exec-order-indexed SoA
//!   (`Vec` per node — no string keys, no `HashMap` on the hot path);
//! * the per-domain kernel invocations are the EXACT same per-layer calls the
//!   scalar reference makes (`propagate_node_ibp_with_engine`,
//!   `propagate_ibp_binary`, `propagate_ibp_ternary`, `propagate_ibp_nary`,
//!   `propagate_ibp_with_condition`), in the exact same exec order, on the
//!   exact same per-domain operands — so every per-domain result is
//!   BIT-IDENTICAL to the reference **by construction**: no arithmetic is
//!   re-implemented, no reduction is re-associated, and the batch dimension
//!   never enters any kernel (parity class: bit-identical; see
//!   `test_batched_interm_bit_identical_to_per_domain_collect`).
//!
//! The per-domain `HashMap<String, BoundedTensor>` is still materialized at
//! the very end (one insert per node per domain, exactly the entries the
//! reference produces) because the downstream batched backward consumes
//! `&HashMap<String, BoundedTensor>` per domain and slice S2 deliberately
//! does NOT touch the backward. Replacing that seam with node-major
//! `[B, W_n]` planes is slice S3's job.
//!
//! # Decline discipline (checklist I-C2 pattern)
//!
//! The batched pass is strictly a performance transform: it only runs for
//! graphs where the scalar reference (`collect_intermediate_bounds`) provably
//! takes the plain-IBP forward arm AND where no bounds-cache-consuming
//! tightening arm (SwiGLU / attention zonotope) can structurally engage —
//! everything else declines to the untouched per-domain reference path.
//! See [`GraphNetwork::batched_interm_forward_supported`].

use crate::layers::Layer;
use crate::network::core::graph::ibp::dispatch::{
    check_nan_firewall, classify_node_inputs, ResolvedInputNames,
};
use crate::network::core::{GraphNetwork, GraphNode, NETWORK_INPUT};

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use super::ibp::propagate_node_ibp_with_engine;

/// Where a node input's bounds come from during the batched forward.
#[derive(Clone, Copy)]
enum SrcIdx {
    /// The network input box (per-domain).
    Input,
    /// An earlier node, by exec-order index into the SoA.
    Node(usize),
}

/// One Concat operand: an embedded constant or a graph edge.
enum ConcatSrc {
    /// Boxed to reduce enum size (clippy::large_enum_variant); built once per
    /// batch prep, read as a single deref per domain.
    Constant(Box<BoundedTensor>),
    Dynamic(SrcIdx),
}

/// Pre-resolved dispatch classification for one node (computed once per
/// batch; replaces the per-domain `classify_node_inputs` + `bounds_ref`
/// string lookups of the scalar reference).
enum PreparedInputs {
    Unary(SrcIdx),
    Binary(SrcIdx, SrcIdx),
    Ternary(SrcIdx, SrcIdx, SrcIdx),
    NaryConcat(Vec<ConcatSrc>),
}

struct PreparedNode<'g> {
    name: &'g str,
    node: &'g GraphNode,
    inputs: PreparedInputs,
}

impl GraphNetwork {
    /// Whether this graph is in the proven class for the batched
    /// intermediate-bounds forward (#lsnc-batched-interm).
    ///
    /// The batched pass replicates the plain-IBP forward arm of
    /// `collect_intermediate_bounds` kernel-for-kernel. It must therefore
    /// decline (fall back to the per-domain reference) whenever the reference
    /// would do anything else:
    ///
    /// * conv layers — the conv-DAG forward-linear arm (with its own cache)
    ///   may run instead of plain IBP;
    /// * per-node CROWN-IBP class (`should_collect_per_node_crown_ibp_intermediates`)
    ///   — the reference runs the O(N²) tightening loop, not plain IBP;
    /// * `MatMul` with `transpose_b` — the attention-zonotope tightening arm
    ///   consumes the full per-domain bounds cache;
    /// * `MulBinary` whose SwiGLU pattern could structurally engage (an input
    ///   is `SiLU`) — the SwiGLU-zonotope arm consumes the bounds cache. For
    ///   every other `MulBinary`, `try_ffn_swiglu_bounds_zonotope` returns
    ///   `None` from its structural prechecks WITHOUT reading any bounds, so
    ///   the plain `propagate_ibp_binary` result is bit-identical.
    ///
    /// MAINTENANCE INVARIANT: if `collect_node_bounds_core` ever gains a new
    /// layer-conditional arm (especially a bounds-cache-consuming tightening
    /// arm), this decline list MUST be extended to cover it in the same
    /// change — otherwise the batched pass silently diverges from the
    /// reference and the bit-parity claim (and its gate default) is void.
    /// The parity test's fixture should gain coverage for the new arm too.
    pub(crate) fn batched_interm_forward_supported(&self) -> bool {
        if self.has_conv_layers() || self.should_collect_per_node_crown_ibp_intermediates() {
            return false;
        }
        for node in self.nodes.values() {
            match &node.layer {
                Layer::MatMul(matmul) if matmul.transpose_b => return false,
                Layer::MulBinary(_) => {
                    let Ok((input_a, input_b)) = node.require_binary_inputs() else {
                        // The reference's swiglu precheck returns None here
                        // without touching bounds; plain IBP arm still runs.
                        continue;
                    };
                    let is_silu = |name: &str| {
                        self.nodes
                            .get(name)
                            .is_some_and(|n| matches!(n.layer, Layer::SiLU(_)))
                    };
                    if is_silu(input_a) || is_silu(input_b) {
                        return false;
                    }
                }
                _ => {}
            }
        }
        true
    }

    /// Resolve the exec order + per-node dispatch once for the whole batch.
    ///
    /// Returns `None` (decline to the reference path) on any structural
    /// resolution problem — the reference will surface the identical error
    /// per domain.
    fn prepare_batched_interm_nodes(&self) -> Option<Vec<PreparedNode<'_>>> {
        let exec_order = self.exec_order().ok()?;
        let mut idx_by_name: HashMap<&str, usize> = HashMap::with_capacity(exec_order.len());
        let mut prepared = Vec::with_capacity(exec_order.len());

        for (idx, node_name) in exec_order.iter().enumerate() {
            let node = self.nodes.get(node_name)?;
            let resolve = |name: &str| -> Option<SrcIdx> {
                if name == NETWORK_INPUT {
                    Some(SrcIdx::Input)
                } else {
                    // Dependency-order violation ⇒ decline; the reference
                    // errors with "not yet computed" per domain.
                    idx_by_name.get(name).copied().map(SrcIdx::Node)
                }
            };
            let classified = classify_node_inputs(node, node_name).ok()?;
            let inputs = match classified {
                ResolvedInputNames::Unary(a) => PreparedInputs::Unary(resolve(a)?),
                ResolvedInputNames::Binary(a, b) => {
                    PreparedInputs::Binary(resolve(a)?, resolve(b)?)
                }
                ResolvedInputNames::Ternary(a, b, c) => {
                    PreparedInputs::Ternary(resolve(a)?, resolve(b)?, resolve(c)?)
                }
                ResolvedInputNames::NaryConcat {
                    dynamic_inputs,
                    has_constants,
                } => {
                    // Mirror the reference's constant/dynamic interleaving
                    // (collect_node_bounds_core NaryConcat arm) exactly.
                    let concat = match &node.layer {
                        Layer::Concat(c) => c,
                        _ => return None,
                    };
                    let srcs: Vec<ConcatSrc> = if has_constants {
                        let ci = concat.constant_inputs.as_ref()?;
                        let mut dyn_idx = 0usize;
                        let mut srcs = Vec::with_capacity(ci.len());
                        for const_opt in ci.iter() {
                            match const_opt {
                                Some(constant) => {
                                    srcs.push(ConcatSrc::Constant(Box::new(constant.clone())))
                                }
                                None => {
                                    let name = dynamic_inputs.get(dyn_idx)?;
                                    dyn_idx += 1;
                                    srcs.push(ConcatSrc::Dynamic(resolve(name)?));
                                }
                            }
                        }
                        srcs
                    } else {
                        dynamic_inputs
                            .iter()
                            .map(|name| resolve(name).map(ConcatSrc::Dynamic))
                            .collect::<Option<Vec<_>>>()?
                    };
                    PreparedInputs::NaryConcat(srcs)
                }
            };
            prepared.push(PreparedNode {
                name: node_name.as_str(),
                node,
                inputs,
            });
            idx_by_name.insert(node_name.as_str(), idx);
        }
        Some(prepared)
    }

    /// Batched multi-domain plain-IBP forward (#lsnc-batched-interm, slice S2).
    ///
    /// For each domain `b`, computes the IDENTICAL per-node bounds map that
    /// `collect_intermediate_bounds(graph, &inputs[b], deadline, engine)`
    /// produces when the graph is in the supported class (plain-IBP forward
    /// arm; see [`Self::batched_interm_forward_supported`]) — same kernels,
    /// same exec order, same NaN-firewall placement, same error text — but
    /// with the graph-structure resolution hoisted out of the per-domain loop
    /// and the rayon granularity coarsened from one task per domain to one
    /// task per domain-chunk.
    ///
    /// Returns `None` when the graph must decline (caller falls back to the
    /// untouched per-domain reference path); otherwise one `Result` per
    /// domain, index-aligned with `inputs` (a failing domain does not poison
    /// its batch-mates, mirroring the reference's per-domain `Result`s).
    ///
    /// Parity class: BIT-IDENTICAL (no re-implemented arithmetic, no
    /// cross-domain state, no reassociated reductions). Parity tests:
    /// `test_batched_interm_bit_identical_to_per_domain_collect`,
    /// `test_batched_interm_permutation_equivariant`.
    pub(crate) fn collect_node_bounds_batched(
        &self,
        inputs: &[BoundedTensor],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Option<Vec<Result<HashMap<String, BoundedTensor>>>> {
        if !self.batched_interm_forward_supported() {
            return None;
        }
        let prepared = self.prepare_batched_interm_nodes()?;
        if inputs.is_empty() {
            return Some(Vec::new());
        }

        // Coarse chunking: a handful of large work items instead of one tiny
        // rayon task per domain (the fan-out the profile showed as idle time).
        let threads = rayon::current_num_threads().max(1);
        let chunk_size = inputs.len().div_ceil(threads * 4).max(1);

        let results: Vec<Result<HashMap<String, BoundedTensor>>> = inputs
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                chunk
                    .iter()
                    .map(|input| {
                        self.collect_node_bounds_one_domain(&prepared, input, engine, deadline)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        Some(results)
    }

    /// One domain's forward over the pre-resolved node list. Kernel calls and
    /// their order mirror `collect_node_bounds_core` (plain-IBP arm,
    /// `allow_negative_sqrt = false`, `collapse_to_center = false`) verbatim.
    fn collect_node_bounds_one_domain(
        &self,
        prepared: &[PreparedNode<'_>],
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        // Exec-order-indexed SoA row for this domain — replaces the
        // reference's String-keyed HashMap on the hot path.
        let mut cache: Vec<BoundedTensor> = Vec::with_capacity(prepared.len());

        for prep in prepared {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(NyError::DeadlineExceeded(format!(
                    "Graph IBP: deadline exceeded before node '{}'",
                    prep.name
                )));
            }
            let fetch = |src: &SrcIdx| -> &BoundedTensor {
                match src {
                    SrcIdx::Input => input,
                    SrcIdx::Node(idx) => &cache[*idx],
                }
            };
            let output_bounds = match &prep.inputs {
                PreparedInputs::Unary(a) => {
                    let bounds = fetch(a);
                    match &prep.node.layer {
                        Layer::Where(w) if w.has_embedded_constants() => {
                            w.propagate_ibp_with_condition(bounds)?
                        }
                        _ => propagate_node_ibp_with_engine(&prep.node.layer, bounds, engine)?,
                    }
                }
                PreparedInputs::Binary(a, b) => {
                    // Supported-class guarantee: the MatMul(transpose_b) and
                    // SwiGLU-MulBinary tightening arms cannot engage (declined
                    // structurally), and for every remaining MulBinary the
                    // reference's `try_ffn_swiglu_bounds_zonotope` returns
                    // `None` before reading any bounds — so plain
                    // `propagate_ibp_binary` is exactly what the reference
                    // computes.
                    prep.node.layer.propagate_ibp_binary(fetch(a), fetch(b))?
                }
                PreparedInputs::Ternary(a, b, c) => {
                    prep.node
                        .layer
                        .propagate_ibp_ternary(fetch(a), fetch(b), fetch(c))?
                }
                PreparedInputs::NaryConcat(srcs) => {
                    let concat = match &prep.node.layer {
                        Layer::Concat(c) => c,
                        _ => {
                            return Err(NyError::InternalError(format!(
                                "batched interm forward: NaryConcat for non-Concat node '{}'",
                                prep.name
                            )));
                        }
                    };
                    let input_refs: Vec<&BoundedTensor> = srcs
                        .iter()
                        .map(|src| match src {
                            ConcatSrc::Constant(c) => c.as_ref(),
                            ConcatSrc::Dynamic(idx) => fetch(idx),
                        })
                        .collect();
                    concat.propagate_ibp_nary(&input_refs)?
                }
            };

            // Same firewall, same position, same context string as the
            // reference (#3768).
            check_nan_firewall(
                &output_bounds,
                "collect_node_bounds",
                prep.name,
                prep.node.layer.layer_type(),
            )?;
            cache.push(output_bounds);
        }

        // Seam adaptation: the downstream batched backward consumes
        // `&HashMap<String, BoundedTensor>` per domain (unchanged in S2), so
        // materialize the reference-identical map here — one insert per node,
        // exactly the reference's entries.
        Ok(prepared
            .iter()
            .map(|p| p.name.to_string())
            .zip(cache)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{
        ConcatLayer, DivLayer, LinearLayer, MulBinaryLayer, ReLULayer, ReduceSumLayer, SiLULayer,
        SubLayer,
    };
    use crate::network::collect_intermediate_bounds;
    use ndarray::{arr1, Array1, Array2};
    use ny_core::NaiveCpuGemmEngine;

    /// Deterministic LCG for fixture data (pattern: relaxed_clip.rs:722).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Uniform in [-1, 1).
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    /// lsnc-shaped fixture: 6-dim input; Linear/ReLU trunk; MulBinary, Sub,
    /// Div binary arms; Concat; ReduceSum. The MulBinary nodes put the graph
    /// in the plain-IBP class (`should_use_crown_ibp_intermediates` is false),
    /// exactly like the real lsnc net. Weights carry mixed signs, exact
    /// zeros, and near-zero (|w| <= 1e-10) entries.
    fn build_lsnc_shaped_graph() -> GraphNetwork {
        let mut lcg = Lcg(0x5eed_15c0);
        let mut w1 = Array2::<f32>::zeros((8, 6));
        for v in w1.iter_mut() {
            *v = lcg.next_f32();
        }
        // Exact zeros + near-zero stripe.
        w1[[0, 0]] = 0.0;
        w1[[1, 1]] = 1e-12;
        w1[[2, 2]] = -1e-12;
        let b1 = Array1::from_iter((0..8).map(|_| lcg.next_f32()));
        let mut w2 = Array2::<f32>::zeros((8, 8));
        for v in w2.iter_mut() {
            *v = lcg.next_f32();
        }
        w2[[3, 3]] = 0.0;
        let b2 = Array1::from_iter((0..8).map(|_| lcg.next_f32()));
        let mut w3 = Array2::<f32>::zeros((4, 16));
        for v in w3.iter_mut() {
            *v = lcg.next_f32();
        }
        let b3 = Array1::from_iter((0..4).map(|_| lcg.next_f32()));

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid linear1")),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("valid linear2")),
            vec!["relu1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "mulb",
            Layer::MulBinary(MulBinaryLayer),
            vec!["relu1".to_string(), "linear2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "subb",
            Layer::Sub(SubLayer),
            vec!["linear2".to_string(), "mulb".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "divb",
            Layer::Div(DivLayer),
            vec!["subb".to_string(), "linear2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "concat1",
            Layer::Concat(ConcatLayer::new(0)),
            vec!["mulb".to_string(), "divb".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear3",
            Layer::Linear(LinearLayer::new(w3, Some(b3)).expect("valid linear3")),
            vec!["concat1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "rsum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![0], true)),
            vec!["linear3".to_string()],
        ));
        graph.set_output("rsum");
        graph
    }

    /// Adversarial domain boxes per the parity checklist: >= 8 domains,
    /// per-domain boxes that flip ReLU stability classes, a degenerate
    /// (zero-width) box, a huge box driving overflow/degenerate kernels.
    fn build_domains() -> Vec<BoundedTensor> {
        let mut lcg = Lcg(0xa11d_0035);
        let mut domains = Vec::new();
        for d in 0..12 {
            let mut lo = [0f32; 6];
            let mut hi = [0f32; 6];
            for i in 0..6 {
                let center = lcg.next_f32() * 2.0;
                let width = match d % 4 {
                    0 => 2.0,  // wide: unstable ReLUs
                    1 => 1e-6, // near-degenerate
                    2 => 0.0,  // exactly degenerate
                    _ => 0.5,  // moderate
                };
                lo[i] = center - width;
                hi[i] = center + width;
            }
            if d == 11 {
                // Overflow probe: enormous box → inf/degenerate downstream.
                lo = [-1e30; 6];
                hi = [1e30; 6];
            }
            domains.push(
                BoundedTensor::new(
                    Array1::from_iter(lo).into_dyn(),
                    Array1::from_iter(hi).into_dyn(),
                )
                .expect("valid domain box"),
            );
        }
        domains
    }

    fn assert_maps_bit_identical(
        reference: &HashMap<String, BoundedTensor>,
        batched: &HashMap<String, BoundedTensor>,
        ctx: &str,
    ) {
        assert_eq!(
            reference.len(),
            batched.len(),
            "{ctx}: node-bounds map sizes differ"
        );
        for (name, ref_bounds) in reference {
            let bat_bounds = batched
                .get(name)
                .unwrap_or_else(|| panic!("{ctx}: node '{name}' missing from batched map"));
            assert_eq!(
                ref_bounds.shape(),
                bat_bounds.shape(),
                "{ctx}: node '{name}' shape differs"
            );
            for (r, b) in ref_bounds.lower().iter().zip(bat_bounds.lower().iter()) {
                assert_eq!(
                    r.to_bits(),
                    b.to_bits(),
                    "{ctx}: node '{name}' LOWER bits differ ({r} vs {b})"
                );
            }
            for (r, b) in ref_bounds.upper().iter().zip(bat_bounds.upper().iter()) {
                assert_eq!(
                    r.to_bits(),
                    b.to_bits(),
                    "{ctx}: node '{name}' UPPER bits differ ({r} vs {b})"
                );
            }
        }
    }

    /// #lsnc-batched-interm parity: the batched collector must be
    /// BIT-IDENTICAL, per domain and per node, to the production per-domain
    /// reference (`collect_intermediate_bounds`) — including identical error
    /// behavior on degenerate domains.
    #[test]
    fn test_batched_interm_bit_identical_to_per_domain_collect() {
        let graph = build_lsnc_shaped_graph();
        assert!(
            graph.batched_interm_forward_supported(),
            "fixture must be in the supported (plain-IBP) class"
        );
        let domains = build_domains();
        let engine = NaiveCpuGemmEngine;

        let batched = graph
            .collect_node_bounds_batched(&domains, Some(&engine), None)
            .expect("supported graph must not decline");
        assert_eq!(batched.len(), domains.len());

        for (b, input) in domains.iter().enumerate() {
            let reference = collect_intermediate_bounds(&graph, input, None, Some(&engine));
            match (&reference, &batched[b]) {
                (Ok(ref_map), Ok(bat_map)) => {
                    assert_maps_bit_identical(ref_map, bat_map, &format!("domain {b}"));
                }
                (Err(ref_err), Err(bat_err)) => {
                    assert_eq!(
                        ref_err.to_string(),
                        bat_err.to_string(),
                        "domain {b}: error text must be identical"
                    );
                }
                (r, bt) => panic!(
                    "domain {b}: result kind mismatch (reference ok={}, batched ok={})",
                    r.is_ok(),
                    bt.is_ok()
                ),
            }
        }
    }

    /// Cross-domain aliasing guard (design-doc soundness risk #2): permuting
    /// the batch must permute the results identically — any b-index mixing
    /// shows up as a bit difference.
    #[test]
    fn test_batched_interm_permutation_equivariant() {
        let graph = build_lsnc_shaped_graph();
        let domains = build_domains();
        let engine = NaiveCpuGemmEngine;

        let forward = graph
            .collect_node_bounds_batched(&domains, Some(&engine), None)
            .expect("supported graph must not decline");

        // Reverse permutation.
        let permuted_domains: Vec<BoundedTensor> = domains.iter().rev().cloned().collect();
        let permuted = graph
            .collect_node_bounds_batched(&permuted_domains, Some(&engine), None)
            .expect("supported graph must not decline");

        let n = domains.len();
        for b in 0..n {
            match (&forward[b], &permuted[n - 1 - b]) {
                (Ok(fwd), Ok(perm)) => {
                    assert_maps_bit_identical(fwd, perm, &format!("permuted domain {b}"));
                }
                (Err(e1), Err(e2)) => assert_eq!(e1.to_string(), e2.to_string()),
                _ => panic!("permuted domain {b}: result kind mismatch"),
            }
        }
    }

    /// Decline leg: a small pure Linear/ReLU chain is in the per-node
    /// CROWN-IBP class, where the reference does NOT run plain IBP — the
    /// batched collector must decline.
    #[test]
    fn test_batched_interm_declines_crown_ibp_class_graph() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(
                LinearLayer::new(Array2::from_elem((2, 2), 1.0), None).expect("valid linear"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.set_output("relu1");
        assert!(
            !graph.batched_interm_forward_supported(),
            "per-node CROWN-IBP class graph must decline"
        );
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .expect("valid box");
        assert!(
            graph
                .collect_node_bounds_batched(&[input], None, None)
                .is_none(),
            "collector must return None (decline) for unsupported graphs"
        );
    }

    /// Decline leg: a structurally SwiGLU-shaped MulBinary (one input SiLU)
    /// could engage the bounds-cache-consuming zonotope arm — must decline.
    #[test]
    fn test_batched_interm_declines_swiglu_pattern() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "gate",
            Layer::Linear(
                LinearLayer::new(Array2::from_elem((2, 2), 0.5), None).expect("valid gate"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "silu",
            Layer::SiLU(SiLULayer),
            vec!["gate".to_string()],
        ));
        graph.add_node(GraphNode::from_input(
            "up",
            Layer::Linear(
                LinearLayer::new(Array2::from_elem((2, 2), -0.5), None).expect("valid up"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            vec!["up".to_string(), "silu".to_string()],
        ));
        graph.set_output("mul");
        assert!(
            !graph.batched_interm_forward_supported(),
            "SwiGLU-pattern MulBinary must decline the batched forward"
        );
    }
}
