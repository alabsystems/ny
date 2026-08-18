// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #extract-skeleton: static/dynamic split of the per-domain resnet segment
//! extraction ([`super::resnet_decompose`]).
//!
//! # Why this exists
//!
//! `extract_gpu_segments_with_relu_names_ext` re-materializes EVERY layer payload
//! (Linear weights, Conv weight-cols, spatially-expanded biases, constant-arith
//! vecs) for EVERY BaB domain, even though only the ReLU relaxations, MaxPool
//! routing, and the abs-max tables actually depend on the domain. This module
//! splits that work: [`build_resnet_segment_skeleton`] runs the legacy backward
//! walk ONCE (on an exemplar domain) capturing everything graph-static, and
//! [`ResnetSegmentSkeleton::fold_for_domain`] re-bakes only the per-domain slots.
//! It is the graph-path twin of the sequential path's `GpuCrownStaticCache`
//! (`network/core/sequential/crown/gpu_extraction.rs`).
//!
//! # Soundness (the 0-wrong moat)
//!
//! 1. **Byte-identity by construction.** The build IS the legacy walk (the same
//!    `extract_gpu_resnet_segments_collect`, with a passive recorder attached),
//!    and the fold re-bakes each dynamic slot through the SAME helpers the legacy
//!    path calls (`bake_relu_layer`, `try_extract_single_gpu_layer`,
//!    `frontier_abs_max_of`, `collect_relu_pre_abs`). Whenever fold and legacy
//!    both succeed they agree bit-for-bit (oracle-enforced in the tests below).
//! 2. **`None`-agreement is behavior.** The fold re-checks every per-domain
//!    refusal the legacy walk performs — active non-ReLU alpha on any visited
//!    node, a missing bounds entry for any resolved input, any slot re-bake
//!    failure, any frontier resolve failure — so it refuses at least whenever
//!    legacy refuses. Extra refusals (stale shape, structural surprises) only
//!    route the increment-2 caller to the legacy per-domain extraction: fail
//!    closed, never a divergent segment list.
//! 3. **Slots are classified by BUILD-TIME ORIGIN (`Layer` variant of the graph
//!    node), never by `GpuCrownLayer` variant** — a constant-arith `Activation`
//!    (AddConstant/SubConstant/MulConstant/DivConstant) is STATIC; a ReLU
//!    `Activation`/`ActivationReluDualAlpha` and a `MaxPool2d` are per-domain.
//!    Bounds-dependent relaxations with no slot kind (Sigmoid/Tanh/Exp/Log) and
//!    any unrecognized layer refuse the BUILD outright.
//! 4. **Dynamic slots are NaN-poisoned in the skeleton.** Cross-domain slot
//!    contamination is the false-VERIFIED hazard (see the slot-contamination
//!    note in ny-gpu `crown_backward_sound_resident.rs`): a leaked unfolded slot
//!    must fail toward the existing NaN-rejection / CPU-fallback, never toward a
//!    finite (wrong) bound. The fold overwrites every slot or returns `None`.
//! 5. **The ReLU layer VARIANT is decided per domain by the fold** (via
//!    `bake_relu_layer`): `Activation` vs `ActivationReluDualAlpha` depends on the
//!    domain's bridged alphas and is never frozen into the skeleton.
//! 6. **`relu_names` fold order** (encounter order, F then P) is captured from
//!    the build walk — the same walk that defines it — so the load-bearing GPU
//!    contract (write_back_alpha / β-gather / node-table slicing) is preserved by
//!    construction.

use std::borrow::Borrow;
use std::collections::HashMap;

use ny_core::{GpuCrownLayer, GpuResnetSegment};
use ny_tensor::BoundedTensor;

use super::resnet_decompose::{
    bake_relu_layer, collect_relu_pre_abs, extract_gpu_resnet_segments_collect,
    frontier_abs_max_of, has_active_non_relu_alpha, resolve_pre,
};
use crate::bounds::GraphAlphaState;
use crate::layers::Layer;
use crate::network::core::{try_extract_single_gpu_layer, GraphNetwork};

/// #extract-skeleton gate — DEFAULT ON (`NY_EXTRACT_SKELETON=0` is the
/// kill-switch reverting to the legacy per-domain extraction wholesale).
///
/// ON routes the batched-BaB prep families (`prep_resnet_domain_with` callers)
/// through build-once-fold-per-domain: [`build_resnet_segment_skeleton`] runs
/// the legacy walk once per call and [`ResnetSegmentSkeleton::fold_for_domain`]
/// re-bakes only the per-domain slots. The fold is oracle-proven bit-identical
/// to `extract_gpu_segments_with_relu_names_ext` whenever both succeed (this
/// module's tests), and EVERY divergence — build refusal, stale skeleton,
/// key mismatch, per-domain fold refusal — falls back to the legacy extraction
/// for that domain, so behavior is identical by construction (the 0-wrong
/// moat). OFF (`=0`) skips the skeleton build entirely: every domain preps
/// through the legacy extraction, byte-identically.
#[inline]
pub(crate) fn extract_skeleton_enabled() -> bool {
    std::env::var("NY_EXTRACT_SKELETON").ok().as_deref() != Some("0")
}

/// Which layer vec of a segment a dynamic slot lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotBranch {
    /// `Chain(v)` / `Residual(f)` / `ResidualProj(f, _)`: the primary vec.
    Main,
    /// `ResidualProj(_, p)`: the projection branch.
    Proj,
}

/// What a dynamic slot re-bakes per domain. Classified by the graph node's
/// `Layer` variant at build time — NEVER by the extracted `GpuCrownLayer`
/// variant (a constant-arith `Activation` is static; see module doc §3).
#[derive(Clone, Debug)]
enum SlotKind {
    /// A ReLU whose relaxation (slopes/intercepts AND `Activation` vs
    /// `ActivationReluDualAlpha` variant) is re-baked from the domain's pre
    /// bounds + bridged alpha via `bake_relu_layer` (module doc §5).
    Relu {
        node: String,
        /// The ReLU's unary input node — its pre-activation bounds source.
        input: String,
    },
    /// A MaxPool2d whose winner routing / IBP-fallback windows are re-extracted
    /// from the domain's pre bounds via `try_extract_single_gpu_layer`.
    MaxPool { node: String, input: String },
}

/// One walk-order dynamic slot: WHERE in the segment structure, and WHAT to
/// re-bake there per domain.
#[derive(Clone, Debug)]
struct DynamicSlot {
    seg_idx: usize,
    branch: SlotBranch,
    layer_idx: usize,
    kind: SlotKind,
}

/// Build-time origin of one extracted layer, recorded by the walk. The
/// static/dynamic decision is made HERE, from the graph `Layer` — the only
/// place the origin is unambiguous (module doc §3).
#[derive(Clone, Debug)]
pub(in crate::network::graph_alpha) enum LayerOrigin {
    /// Linear/Conv weight layer: `Arc<[f32]>` payload from the layer itself,
    /// geometry from the (shape-guarded) pre-activation shape — graph-static.
    StaticWeight,
    /// Constant-arith `Activation` (AddConstant/SubConstant/MulConstant/
    /// DivConstant): payload from the layer's constant — graph-static even
    /// though its VARIANT matches the dynamic ReLU relaxation's.
    StaticConstArith,
    Relu {
        node: String,
        input: String,
    },
    MaxPool {
        node: String,
        input: String,
    },
}

/// Passive recorder the BUILD threads through the legacy collect walk
/// (`extract_gpu_resnet_segments_collect(.., Some(&mut rec))`). Every recording
/// hook in the walk is `if let Some(..)`-guarded, so legacy/production calls
/// (`rec = None`) are byte-identical to the pre-#extract-skeleton code.
/// Recording can only add a REFUSAL (an unclassifiable layer fails the build),
/// never change an extracted segment.
#[derive(Default)]
pub(in crate::network::graph_alpha) struct SkeletonRecorder {
    /// Origins for the layer vec currently being accumulated (the main loop's
    /// `chain`, or the branch being walked by `extract_unary_path_to_z` —
    /// never both at once: the chain is always flushed before a block walk).
    pending: Vec<LayerOrigin>,
    /// A decomposed residual block's (F, P) origins, staged by
    /// `decompose_residual_block` and committed by the collect loop once the
    /// block's final segment index is known.
    staged_block: Option<(Vec<LayerOrigin>, Vec<LayerOrigin>)>,
    /// Walk-order dynamic slots with final positions.
    dynamic_slots: Vec<DynamicSlot>,
    /// Per segment, the input-side frontier node name (`NETWORK_INPUT` sentinel
    /// for the final chain) — the fold re-derives `frontier_abs` from these.
    frontier_nodes: Vec<String>,
    /// Every node the walk visited — the fold re-runs the per-domain
    /// `has_active_non_relu_alpha` refusal over exactly this set.
    visited_nodes: Vec<String>,
    /// Every `(input name, shape)` the walk `resolve_pre`'d — the fold refuses
    /// on a missing entry (legacy `None`-agreement) or a changed shape
    /// (stale-skeleton fail-closed guard; module doc §2).
    resolved_inputs: Vec<(String, Vec<usize>)>,
    /// Count of static constant-arith `Activation` layers, for the build-time
    /// variant-vs-origin structural cross-check.
    static_const_arith: usize,
}

impl SkeletonRecorder {
    pub(in crate::network::graph_alpha) fn record_visited(&mut self, node: &str) {
        self.visited_nodes.push(node.to_string());
    }

    pub(in crate::network::graph_alpha) fn record_resolved(
        &mut self,
        name: &str,
        bt: &BoundedTensor,
    ) {
        self.resolved_inputs
            .push((name.to_string(), bt.shape().to_vec()));
    }

    /// Classify ONE node's extraction (`pushed` = layers it appended) by its
    /// build-time origin. `None` refuses the BUILD (→ the caller keeps using
    /// legacy per-domain extraction — fail closed): bounds-dependent
    /// relaxations without a slot kind (Sigmoid/Tanh/Exp/Log), any layer kind
    /// this classification does not know, or an unexpected push count.
    pub(in crate::network::graph_alpha) fn record_layer(
        &mut self,
        layer: &Layer,
        node: &str,
        input: &str,
        pushed: usize,
    ) -> Option<()> {
        if pushed == 0 {
            // Flatten/Reshape are no-ops for flat A-matrices — no layer, no slot.
            return matches!(layer, Layer::Flatten(_) | Layer::Reshape(_)).then_some(());
        }
        if pushed != 1 {
            return None;
        }
        let origin = match layer {
            Layer::ReLU(_) => LayerOrigin::Relu {
                node: node.to_string(),
                input: input.to_string(),
            },
            Layer::MaxPool2d(_) => LayerOrigin::MaxPool {
                node: node.to_string(),
                input: input.to_string(),
            },
            Layer::Linear(_) | Layer::Conv1d(_) | Layer::Conv2d(_) => LayerOrigin::StaticWeight,
            Layer::AddConstant(_)
            | Layer::SubConstant(_)
            | Layer::MulConstant(_)
            | Layer::DivConstant(_) => LayerOrigin::StaticConstArith,
            // Sigmoid/Tanh/Exp/Log relaxations depend on the exemplar domain's
            // bounds and have no re-bake slot kind; a future layer kind is
            // unclassifiable by definition. Refuse the build — never capture a
            // bounds-dependent payload as static.
            _ => return None,
        };
        self.pending.push(origin);
        Some(())
    }

    /// Drain the pending (branch) origins — `decompose_residual_block` calls
    /// this after each branch walk so F's origins survive the P walk.
    pub(in crate::network::graph_alpha) fn take_pending(&mut self) -> Vec<LayerOrigin> {
        std::mem::take(&mut self.pending)
    }

    /// Stage a decomposed block's branch origins until the collect loop knows
    /// the block's segment index.
    pub(in crate::network::graph_alpha) fn stage_block(
        &mut self,
        f: Vec<LayerOrigin>,
        p: Vec<LayerOrigin>,
    ) {
        self.staged_block = Some((f, p));
    }

    /// Commit the pending chain origins as segment `seg_idx` (a `Chain`).
    pub(in crate::network::graph_alpha) fn commit_chain(&mut self, seg_idx: usize, frontier: &str) {
        let origins = std::mem::take(&mut self.pending);
        self.frontier_nodes.push(frontier.to_string());
        self.commit_origins(seg_idx, SlotBranch::Main, origins);
    }

    /// Commit the staged block origins as segment `seg_idx` (a `Residual` /
    /// `ResidualProj`). A missing stage yields zero slots here and is caught by
    /// the build's variant-vs-origin cross-check (which then refuses).
    pub(in crate::network::graph_alpha) fn commit_block(&mut self, seg_idx: usize, frontier: &str) {
        let (f, p) = self.staged_block.take().unwrap_or_default();
        self.frontier_nodes.push(frontier.to_string());
        self.commit_origins(seg_idx, SlotBranch::Main, f);
        self.commit_origins(seg_idx, SlotBranch::Proj, p);
    }

    fn commit_origins(&mut self, seg_idx: usize, branch: SlotBranch, origins: Vec<LayerOrigin>) {
        for (layer_idx, origin) in origins.into_iter().enumerate() {
            let kind = match origin {
                LayerOrigin::StaticWeight => continue,
                LayerOrigin::StaticConstArith => {
                    self.static_const_arith += 1;
                    continue;
                }
                LayerOrigin::Relu { node, input } => SlotKind::Relu { node, input },
                LayerOrigin::MaxPool { node, input } => SlotKind::MaxPool { node, input },
            };
            self.dynamic_slots.push(DynamicSlot {
                seg_idx,
                branch,
                layer_idx,
                kind,
            });
        }
    }

    /// Everything committed (no origins stranded outside a segment)?
    fn is_drained(&self) -> bool {
        self.pending.is_empty() && self.staged_block.is_none()
    }
}

/// The static half of a resnet-suffix extraction: segment structure with all
/// graph-static payloads baked (`Arc<[f32]>` weights → O(1)/layer clone) and
/// NaN-POISONED placeholders in every per-domain slot (module doc §4), plus the
/// walk metadata the per-domain [`Self::fold_for_domain`] needs.
pub(crate) struct ResnetSegmentSkeleton {
    segments: Vec<GpuResnetSegment>,
    /// Walk-order dynamic slots to re-bake per domain.
    dynamic_slots: Vec<DynamicSlot>,
    /// Fold-order per-ReLU node names (encounter order, F then P — the
    /// load-bearing GPU contract; module doc §6).
    relu_names: Vec<String>,
    /// Per segment, the input-side frontier node (`NETWORK_INPUT` for the last).
    frontier_nodes: Vec<String>,
    /// Per-domain `has_active_non_relu_alpha` re-check set.
    visited_nodes: Vec<String>,
    /// Per-domain missing-bounds refusal parity + stale-shape guard set.
    resolved_inputs: Vec<(String, Vec<usize>)>,
    start_node: String,
    allow_pure_chain: bool,
    graph_nodes_len: usize,
    /// Node-name sequence for [`Self::matches_graph`] staleness validation.
    graph_node_order: Vec<String>,
}

/// Build a [`ResnetSegmentSkeleton`] by running the LEGACY backward walk once on
/// an exemplar domain with the build recorder attached, then poisoning every
/// dynamic slot. Returns `None` whenever the legacy extraction would refuse this
/// exemplar, AND on any skeleton-specific refusal (unclassifiable layer,
/// recorder inconsistency, failed variant-vs-origin cross-check) — the caller
/// then stays on legacy per-domain extraction (fail closed).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_resnet_segment_skeleton<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    start_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    allow_pure_chain: bool,
) -> Option<ResnetSegmentSkeleton> {
    let mut relu_names = Vec::new();
    let mut frontier_abs = Vec::new();
    let mut rec = SkeletonRecorder::default();
    // #extract-skeleton x #image-node-crown: the build ALWAYS records the legacy
    // walk — `allow_bn`/`frozen_stop` hard-false (the collect refuses a recorder
    // combined with either flag, and `prep_resnet_domain_with` declines the
    // skeleton fold entirely whenever a caller sets one of them).
    let mut stopped_at: Option<String> = None;
    let mut segments = extract_gpu_resnet_segments_collect(
        graph,
        input,
        start_node,
        crown_bounds,
        ibp_bounds,
        alpha_state,
        &mut relu_names,
        &mut frontier_abs,
        allow_pure_chain,
        Some(&mut rec),
        false,
        false,
        &mut stopped_at,
    )?;
    debug_assert!(stopped_at.is_none(), "frozen_stop=false can never stop");
    // Recorder hygiene: every origin must have been committed to a segment, and
    // the frontier-name list must mirror the legacy `frontier_abs` 1:1.
    if !rec.is_drained() || rec.frontier_nodes.len() != segments.len() {
        return None;
    }

    // Structural soundness gate (module doc §3/§4): the variant view and the
    // origin view of "which layers are per-domain" must agree EXACTLY —
    //   #ReLU slots == #fold-order relu names,
    //   #MaxPool2d layers == #MaxPool slots,
    //   #Activation-family layers == #ReLU slots + #const-arith statics.
    // A missed slot would ship one domain's relaxation as another's (the
    // false-VERIFIED hazard), so any mismatch refuses the build.
    let relu_slots = rec
        .dynamic_slots
        .iter()
        .filter(|s| matches!(s.kind, SlotKind::Relu { .. }))
        .count();
    let maxpool_slots = rec.dynamic_slots.len() - relu_slots;
    let mut activation_layers = 0usize;
    let mut maxpool_layers = 0usize;
    for_each_layer(&segments, |layer| match layer {
        GpuCrownLayer::Activation { .. } | GpuCrownLayer::ActivationReluDualAlpha { .. } => {
            activation_layers += 1;
        }
        GpuCrownLayer::MaxPool2d { .. } => maxpool_layers += 1,
        _ => {}
    });
    if relu_slots != relu_names.len()
        || maxpool_layers != maxpool_slots
        || activation_layers != relu_slots + rec.static_const_arith
    {
        return None;
    }

    // NaN-poison every dynamic slot: the exemplar domain's relaxation must not
    // survive in the skeleton (module doc §4). `None` on any placement surprise.
    for slot in &rec.dynamic_slots {
        let layer = slot_layer_mut(&mut segments, slot)?;
        *layer = poisoned_placeholder(layer)?;
    }

    Some(ResnetSegmentSkeleton {
        segments,
        dynamic_slots: rec.dynamic_slots,
        relu_names,
        frontier_nodes: rec.frontier_nodes,
        visited_nodes: rec.visited_nodes,
        resolved_inputs: rec.resolved_inputs,
        start_node: start_node.to_string(),
        allow_pure_chain,
        graph_nodes_len: graph.nodes.len(),
        graph_node_order: graph.node_order.clone(),
    })
}

impl ResnetSegmentSkeleton {
    /// Staleness guard for skeleton reuse (the `alpha_prime_matches` precedent):
    /// the skeleton's conv geometry / constant broadcasts were baked from THIS
    /// graph's bounds shapes, so a cache hit must re-validate the node-name
    /// sequence, node count, and start node before folding. O(#nodes) string
    /// compares; a mismatch means rebuild (increment 3's cache contract).
    pub(crate) fn matches_graph(&self, graph: &GraphNetwork) -> bool {
        graph.nodes.len() == self.graph_nodes_len
            && graph.node_order == self.graph_node_order
            && graph.nodes.contains_key(self.start_node.as_str())
    }

    /// `(start_node, allow_pure_chain)` — the increment-3 cross-batch cache key.
    pub(crate) fn cache_key(&self) -> (&str, bool) {
        (&self.start_node, self.allow_pure_chain)
    }

    /// Fold this skeleton for ONE domain: re-run the per-domain refusals, clone
    /// the static segments (Arc-cheap), re-bake every dynamic slot from THIS
    /// domain's bounds/alpha, and re-derive the abs-max tables. Returns the same
    /// `(segments, relu_names, frontier_abs, node_abs)` tuple as
    /// `extract_gpu_segments_with_relu_names_ext`, bit-identical to it whenever
    /// both succeed; `None` → the caller falls back to the legacy per-domain
    /// extraction (fail closed).
    ///
    /// Caller contract: `graph` is the graph this skeleton was built from
    /// (validate with [`Self::matches_graph`] on any cached reuse).
    pub(crate) fn fold_for_domain<V1: Borrow<BoundedTensor>, V2: Borrow<BoundedTensor>>(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        crown_bounds: &HashMap<String, V1>,
        ibp_bounds: &HashMap<String, V2>,
        alpha_state: Option<&GraphAlphaState>,
    ) -> Option<(
        Vec<GpuResnetSegment>,
        Vec<String>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
    )> {
        // Per-domain refusal parity with the legacy walk (module doc §2,
        // `None`-agreement IS behavior):
        //  (a) an active NON-ReLU alpha on any node the build walk visited;
        for name in &self.visited_nodes {
            if has_active_non_relu_alpha(name, alpha_state) {
                return None;
            }
        }
        //  (b) a missing bounds entry for any input the walk resolves (legacy
        //      `resolve_pre`s EVERY unary node's input, static or dynamic). The
        //      extra SHAPE equality is the stale-skeleton fail-closed guard: a
        //      same-named node with a different bounds shape would silently
        //      mis-shape the conv geometry / constant broadcasts baked at build
        //      time, so refuse (→ legacy fallback) instead of diverging.
        for (name, shape) in &self.resolved_inputs {
            let bt = resolve_pre(input, name, crown_bounds, ibp_bounds)?;
            if bt.shape() != shape.as_slice() {
                return None;
            }
        }

        // Clone the static skeleton (Arc payloads → O(1)/layer) and re-bake
        // every dynamic slot. Every slot is overwritten or the fold refuses —
        // a NaN placeholder can only leak on a bug, and then it fails toward
        // NaN-rejection/CPU-fallback, never a bound (module doc §4).
        let mut segments = self.segments.clone();
        for slot in &self.dynamic_slots {
            let baked = match &slot.kind {
                SlotKind::Relu {
                    node,
                    input: pre_name,
                } => {
                    let pre = resolve_pre(input, pre_name, crown_bounds, ibp_bounds)?;
                    // Variant decided per domain, exactly as legacy (module doc §5).
                    bake_relu_layer(node, pre, alpha_state)?
                }
                SlotKind::MaxPool {
                    node,
                    input: pre_name,
                } => {
                    let pre = resolve_pre(input, pre_name, crown_bounds, ibp_bounds)?;
                    let layer = &graph.nodes.get(node.as_str())?.layer;
                    if !matches!(layer, Layer::MaxPool2d(_)) {
                        // Stale graph slipped past the caller — fail closed.
                        return None;
                    }
                    let mut out = Vec::with_capacity(1);
                    try_extract_single_gpu_layer(layer, pre, &mut out)?;
                    if out.len() != 1 {
                        return None;
                    }
                    out.pop()?
                }
            };
            *slot_layer_mut(&mut segments, slot)? = baked;
        }

        // Per-domain abs-max tables through the IDENTICAL legacy helpers:
        // frontier refusal on any unresolvable frontier node (as legacy), and
        // node_abs' empty-on-unresolvable degradation (as legacy).
        let mut frontier_abs = Vec::with_capacity(self.frontier_nodes.len());
        for name in &self.frontier_nodes {
            frontier_abs.push(frontier_abs_max_of(input, name, crown_bounds, ibp_bounds)?);
        }
        let node_abs =
            collect_relu_pre_abs(graph, input, &self.relu_names, crown_bounds, ibp_bounds)
                .unwrap_or_default();

        Some((segments, self.relu_names.clone(), frontier_abs, node_abs))
    }
}

/// The layer a slot points at, or `None` if the slot does not fit the segment
/// structure (a recorder bug — callers refuse, fail closed).
fn slot_layer_mut<'s>(
    segments: &'s mut [GpuResnetSegment],
    slot: &DynamicSlot,
) -> Option<&'s mut GpuCrownLayer> {
    let seg = segments.get_mut(slot.seg_idx)?;
    let layers = match (seg, slot.branch) {
        (GpuResnetSegment::Chain(v) | GpuResnetSegment::Residual(v), SlotBranch::Main) => v,
        (GpuResnetSegment::ResidualProj(f, _), SlotBranch::Main) => f,
        (GpuResnetSegment::ResidualProj(_, p), SlotBranch::Proj) => p,
        // Chain/Residual carry no projection branch.
        (_, SlotBranch::Proj) => return None,
    };
    layers.get_mut(slot.layer_idx)
}

/// The NaN-poisoned placeholder for a dynamic slot, shaped like the exemplar's
/// extracted layer so a (bug-only) leak keeps dimensional structure but can
/// only produce NaN — the sound failure direction (module doc §4). `None` for
/// a static variant in a dynamic slot: that is a classification bug and must
/// refuse the build.
fn poisoned_placeholder(extracted: &GpuCrownLayer) -> Option<GpuCrownLayer> {
    match extracted {
        GpuCrownLayer::Activation { num_neurons, .. }
        | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => {
            let n = *num_neurons;
            Some(GpuCrownLayer::Activation {
                lower_slope: vec![f32::NAN; n],
                upper_slope: vec![f32::NAN; n],
                lower_intercept: vec![f32::NAN; n],
                upper_intercept: vec![f32::NAN; n],
                num_neurons: n,
            })
        }
        GpuCrownLayer::MaxPool2d {
            routing,
            ibp_lower,
            ibp_upper,
            input_dim,
            output_dim,
        } => Some(GpuCrownLayer::MaxPool2d {
            // All-IBP routing + NaN window bounds: a leaked slot contributes NaN
            // bias, never a finite (wrong) coefficient route.
            routing: vec![u32::MAX; routing.len()],
            ibp_lower: vec![f32::NAN; ibp_lower.len()],
            ibp_upper: vec![f32::NAN; ibp_upper.len()],
            input_dim: *input_dim,
            output_dim: *output_dim,
        }),
        // Linear/Conv2d are static by construction — refuse.
        _ => None,
    }
}

/// Visit every layer of every segment (Chain/Residual main vec, then
/// ResidualProj F then P — the fold order).
fn for_each_layer<'a>(segments: &'a [GpuResnetSegment], mut f: impl FnMut(&'a GpuCrownLayer)) {
    for seg in segments {
        match seg {
            GpuResnetSegment::Chain(v) | GpuResnetSegment::Residual(v) => {
                v.iter().for_each(&mut f);
            }
            GpuResnetSegment::ResidualProj(fb, pb) => {
                fb.iter().for_each(&mut f);
                pb.iter().for_each(&mut f);
            }
        }
    }
}

/// Shared test fixtures + bit-identity comparators for the #extract-skeleton
/// oracles. `pub(crate)` so the dag-alpha warmup call-site oracles
/// (#root-alpha-gpu, `propagate_dag/gradients`) reuse the SAME real-shaped
/// resnet fixtures and the same "every tuple field bit-identical" comparators
/// instead of drifting copies.
#[cfg(test)]
pub(crate) mod test_support {
    use super::super::resnet_decompose::extract_gpu_segments_with_relu_names_ext;
    use super::*;
    use crate::layers::{
        AddConstantLayer, AddLayer, Conv2dLayer, LinearLayer, MaxPool2dLayer, ReLULayer,
    };
    use crate::network::core::{GraphNode, NETWORK_INPUT};
    use ndarray::{arr1, arr2, Array, ArrayD, IxDyn};

    pub(crate) fn box_input(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(shape), lo),
            ArrayD::from_elem(IxDyn(shape), hi),
        )
        .expect("valid input box")
    }

    /// Deterministic LCG in [-1, 1) (the ny-gpu differential-oracle pattern).
    pub(crate) fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    pub(crate) fn conv(
        rng: &mut impl FnMut() -> f32,
        name: &str,
        input: &str,
        (ic, oc): (usize, usize),
        k: usize,
        s: usize,
        p: usize,
        bias: bool,
    ) -> GraphNode {
        let kernel = Array::from_shape_vec(
            IxDyn(&[oc, ic, k, k]),
            (0..oc * ic * k * k).map(|_| rng() * 0.35).collect(),
        )
        .expect("kernel");
        let b = bias.then(|| arr1(&(0..oc).map(|_| rng() * 0.1).collect::<Vec<f32>>()));
        let layer = Layer::Conv2d(Conv2dLayer::new(kernel, b, (s, s), (p, p)).expect("conv"));
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }

    pub(crate) fn relu(name: &str, input: &str) -> GraphNode {
        GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
    }

    pub(crate) fn add(name: &str, a: &str, b: &str) -> GraphNode {
        GraphNode::new(
            name,
            Layer::Add(AddLayer),
            vec![a.to_string(), b.to_string()],
        )
    }

    /// Real-shaped conv resnet exercising every slot/static kind:
    ///
    /// ```text
    /// input[2,6,6] → conv0 → relu0 → maxpool[4,3,3]
    ///   → { b1c1 → b1r1 → b1c2 } + maxpool = add1        (identity skip, z=maxpool)
    ///   → addc (AddConstant: STATIC const-arith Activation)
    ///   → { b2r1 → b2c1 } + { p2c1 } = add2              (projection skip, z=addc)
    ///   → relu_out → conv_out[2,3,3]
    /// ```
    ///
    /// Expected decomposition from `conv_out`:
    /// `[Chain(conv_out,relu_out), ResidualProj([b2c1,b2r1],[p2c1]), Chain(addc),
    ///   Residual([b1c2,b1r1,b1c1]), Chain(maxpool,relu0,conv0)]`,
    /// fold-order relu names `[relu_out, b2r1, b1r1, relu0]`.
    pub(crate) fn conv_resnet_fixture() -> GraphNetwork {
        let mut rng = lcg(0xC4A1_57AC_71F3);
        let mut g = GraphNetwork::new();
        g.add_node(conv(
            &mut rng,
            "conv0",
            NETWORK_INPUT,
            (2, 4),
            3,
            1,
            1,
            true,
        ));
        g.add_node(relu("relu0", "conv0"));
        g.add_node(GraphNode::new(
            "maxpool",
            Layer::MaxPool2d(MaxPool2dLayer::new((2, 2), (2, 2), (0, 0))),
            vec!["relu0".to_string()],
        ));
        g.add_node(conv(&mut rng, "b1c1", "maxpool", (4, 4), 3, 1, 1, false));
        g.add_node(relu("b1r1", "b1c1"));
        g.add_node(conv(&mut rng, "b1c2", "b1r1", (4, 4), 3, 1, 1, true));
        g.add_node(add("add1", "b1c2", "maxpool"));
        g.add_node(GraphNode::new(
            "addc",
            Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[1]), 0.1))),
            vec!["add1".to_string()],
        ));
        g.add_node(relu("b2r1", "addc"));
        g.add_node(conv(&mut rng, "b2c1", "b2r1", (4, 8), 3, 1, 1, true));
        g.add_node(conv(&mut rng, "p2c1", "addc", (4, 8), 1, 1, 0, false));
        g.add_node(add("add2", "b2c1", "p2c1"));
        g.add_node(relu("relu_out", "add2"));
        g.add_node(conv(
            &mut rng,
            "conv_out",
            "relu_out",
            (8, 2),
            1,
            1,
            0,
            true,
        ));
        g.set_output("conv_out");
        g
    }

    pub(crate) const CONV_FIXTURE_RELUS: [&str; 4] = ["relu0", "b1r1", "b2r1", "relu_out"];

    /// Synthetic per-domain alpha bridge: `add_relu_node` from this domain's pre
    /// bounds (the `build_alpha_bridge` recipe), then uniform lower/upper alpha
    /// values — `lo != hi` makes unstable neurons take the DualAlpha variant.
    pub(crate) fn mk_alpha(
        graph: &GraphNetwork,
        bounds: &HashMap<String, BoundedTensor>,
        relus: &[&str],
        lo: f32,
        hi: f32,
    ) -> GraphAlphaState {
        let mut ga = GraphAlphaState::new();
        for name in relus {
            let pre_name = graph
                .nodes
                .get(*name)
                .expect("fixture relu")
                .inputs
                .first()
                .expect("relu input")
                .clone();
            let pre = bounds.get(&pre_name).expect("pre bounds");
            ga.add_relu_node(name, pre, false).expect("add relu node");
            if let Some((l, u)) = ga.relu_alpha_pair_mut(name) {
                l.fill(lo);
                u.fill(hi);
            }
        }
        ga
    }

    // ---------- bit-identity helpers ----------

    pub(crate) fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    pub(crate) fn assert_layer_bits_eq(a: &GpuCrownLayer, b: &GpuCrownLayer, ctx: &str) {
        use GpuCrownLayer as L;
        match (a, b) {
            (
                L::Linear {
                    weight: wa,
                    bias: ba,
                    out_features: oa,
                    in_features: ia,
                    cert_err: cea,
                },
                L::Linear {
                    weight: wb,
                    bias: bb,
                    out_features: ob,
                    in_features: ib,
                    cert_err: ceb,
                },
            ) => {
                assert_eq!(bits(wa), bits(wb), "{ctx}: Linear weight bits");
                assert_eq!(
                    ba.as_deref().map(bits),
                    bb.as_deref().map(bits),
                    "{ctx}: Linear bias bits"
                );
                assert_eq!((oa, ia), (ob, ib), "{ctx}: Linear dims");
                // #cert-err: a rebuild that DROPPED the declared BN-fold error
                // would silently produce a looser-weight/tighter-bound skeleton,
                // so it is part of the bit-equality contract, not an extra.
                assert_eq!(cea, ceb, "{ctx}: Linear cert_err");
            }
            (
                L::Activation {
                    lower_slope: lsa,
                    upper_slope: usa,
                    lower_intercept: lia,
                    upper_intercept: uia,
                    num_neurons: na,
                },
                L::Activation {
                    lower_slope: lsb,
                    upper_slope: usb,
                    lower_intercept: lib,
                    upper_intercept: uib,
                    num_neurons: nb,
                },
            ) => {
                assert_eq!(bits(lsa), bits(lsb), "{ctx}: Activation lower_slope");
                assert_eq!(bits(usa), bits(usb), "{ctx}: Activation upper_slope");
                assert_eq!(bits(lia), bits(lib), "{ctx}: Activation lower_intercept");
                assert_eq!(bits(uia), bits(uib), "{ctx}: Activation upper_intercept");
                assert_eq!(na, nb, "{ctx}: Activation num_neurons");
            }
            (
                L::ActivationReluDualAlpha {
                    lower_pos_slope: lpa,
                    cross_slope: csa,
                    upper_neg_slope: una,
                    cross_intercept: cia,
                    num_neurons: na,
                },
                L::ActivationReluDualAlpha {
                    lower_pos_slope: lpb,
                    cross_slope: csb,
                    upper_neg_slope: unb,
                    cross_intercept: cib,
                    num_neurons: nb,
                },
            ) => {
                assert_eq!(bits(lpa), bits(lpb), "{ctx}: DualAlpha lower_pos_slope");
                assert_eq!(bits(csa), bits(csb), "{ctx}: DualAlpha cross_slope");
                assert_eq!(bits(una), bits(unb), "{ctx}: DualAlpha upper_neg_slope");
                assert_eq!(bits(cia), bits(cib), "{ctx}: DualAlpha cross_intercept");
                assert_eq!(na, nb, "{ctx}: DualAlpha num_neurons");
            }
            (
                L::Conv2d {
                    weight_col: wa,
                    bias_expanded: ba,
                    out_channels: oca,
                    in_channels: ica,
                    kernel_h: kha,
                    kernel_w: kwa,
                    stride_h: sha,
                    stride_w: swa,
                    pad_h: pha,
                    pad_w: pwa,
                    out_h: oha,
                    out_w: owa,
                    in_h: iha,
                    in_w: iwa,
                    cert_err: cea,
                },
                L::Conv2d {
                    weight_col: wb,
                    bias_expanded: bb,
                    out_channels: ocb,
                    in_channels: icb,
                    kernel_h: khb,
                    kernel_w: kwb,
                    stride_h: shb,
                    stride_w: swb,
                    pad_h: phb,
                    pad_w: pwb,
                    out_h: ohb,
                    out_w: owb,
                    in_h: ihb,
                    in_w: iwb,
                    cert_err: ceb,
                },
            ) => {
                assert_eq!(bits(wa), bits(wb), "{ctx}: Conv2d weight_col bits");
                assert_eq!(
                    ba.as_deref().map(bits),
                    bb.as_deref().map(bits),
                    "{ctx}: Conv2d bias_expanded bits"
                );
                assert_eq!(
                    (oca, ica, kha, kwa, sha, swa, pha, pwa, oha, owa, iha, iwa),
                    (ocb, icb, khb, kwb, shb, swb, phb, pwb, ohb, owb, ihb, iwb),
                    "{ctx}: Conv2d geometry"
                );
                // #cert-err: see the Linear arm — part of the bit contract.
                assert_eq!(cea, ceb, "{ctx}: Conv2d cert_err");
            }
            (
                L::MaxPool2d {
                    routing: ra,
                    ibp_lower: la,
                    ibp_upper: ua,
                    input_dim: ida,
                    output_dim: oda,
                },
                L::MaxPool2d {
                    routing: rb,
                    ibp_lower: lb,
                    ibp_upper: ub,
                    input_dim: idb,
                    output_dim: odb,
                },
            ) => {
                assert_eq!(ra, rb, "{ctx}: MaxPool2d routing");
                assert_eq!(bits(la), bits(lb), "{ctx}: MaxPool2d ibp_lower bits");
                assert_eq!(bits(ua), bits(ub), "{ctx}: MaxPool2d ibp_upper bits");
                assert_eq!((ida, oda), (idb, odb), "{ctx}: MaxPool2d dims");
            }
            _ => panic!("{ctx}: layer VARIANT mismatch"),
        }
    }

    pub(crate) fn assert_segments_bits_eq(
        a: &[GpuResnetSegment],
        b: &[GpuResnetSegment],
        ctx: &str,
    ) {
        assert_eq!(a.len(), b.len(), "{ctx}: segment count");
        for (i, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
            match (sa, sb) {
                (GpuResnetSegment::Chain(va), GpuResnetSegment::Chain(vb))
                | (GpuResnetSegment::Residual(va), GpuResnetSegment::Residual(vb)) => {
                    assert_eq!(va.len(), vb.len(), "{ctx}: segment {i} layer count");
                    for (j, (la, lb)) in va.iter().zip(vb.iter()).enumerate() {
                        assert_layer_bits_eq(la, lb, &format!("{ctx}: seg {i} layer {j}"));
                    }
                }
                (
                    GpuResnetSegment::ResidualProj(fa, pa),
                    GpuResnetSegment::ResidualProj(fb, pb),
                ) => {
                    assert_eq!(fa.len(), fb.len(), "{ctx}: seg {i} F layer count");
                    assert_eq!(pa.len(), pb.len(), "{ctx}: seg {i} P layer count");
                    for (j, (la, lb)) in fa.iter().zip(fb.iter()).enumerate() {
                        assert_layer_bits_eq(la, lb, &format!("{ctx}: seg {i} F layer {j}"));
                    }
                    for (j, (la, lb)) in pa.iter().zip(pb.iter()).enumerate() {
                        assert_layer_bits_eq(la, lb, &format!("{ctx}: seg {i} P layer {j}"));
                    }
                }
                _ => panic!("{ctx}: segment {i} VARIANT mismatch"),
            }
        }
    }

    pub(crate) fn assert_tables_bits_eq(a: &[Vec<f32>], b: &[Vec<f32>], ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: table count");
        for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(bits(ra), bits(rb), "{ctx}: table row {i} bits");
        }
    }

    pub(crate) type Extraction = (
        Vec<GpuResnetSegment>,
        Vec<String>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
    );

    /// #extract-skeleton x #image-node-crown: the legacy extraction with the
    /// merged API's new flags hard-false (allow_bn=false, frozen_stop=false) —
    /// the ONLY configuration the skeleton fold may compare against. Asserts the
    /// 5th slot's contract (`stop_node` is None whenever frozen_stop=false) and
    /// drops it so the 4-tuple oracles keep their historical shape.
    pub(crate) fn legacy_ext(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        start: &str,
        crown: &HashMap<String, BoundedTensor>,
        ibp: &HashMap<String, BoundedTensor>,
        alpha: Option<&GraphAlphaState>,
        allow_pure_chain: bool,
    ) -> Option<Extraction> {
        let (segs, names, frontier, node_abs, stop) = extract_gpu_segments_with_relu_names_ext(
            graph,
            input,
            start,
            crown,
            ibp,
            alpha,
            allow_pure_chain,
            false,
            false,
        )?;
        assert!(stop.is_none(), "frozen_stop=false must never stop");
        Some((segs, names, frontier, node_abs))
    }

    /// The increment's oracle: every field of the extraction tuple BIT-identical.
    pub(crate) fn assert_extraction_bits_eq(fold: &Extraction, legacy: &Extraction, ctx: &str) {
        assert_segments_bits_eq(&fold.0, &legacy.0, ctx);
        assert_eq!(fold.1, legacy.1, "{ctx}: relu_names");
        assert_tables_bits_eq(&fold.2, &legacy.2, &format!("{ctx}: frontier_abs"));
        assert_tables_bits_eq(&fold.3, &legacy.3, &format!("{ctx}: node_abs"));
    }

    /// Data pointers of every static Arc payload, in fold order (for the
    /// cross-fold `Arc` sharing assertion).
    pub(crate) fn static_arc_ptrs(segments: &[GpuResnetSegment]) -> Vec<*const f32> {
        let mut out = Vec::new();
        for_each_layer(segments, |l| match l {
            GpuCrownLayer::Linear { weight, bias, .. } => {
                out.push(weight.as_ptr());
                if let Some(b) = bias.as_deref() {
                    out.push(b.as_ptr());
                }
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                ..
            } => {
                out.push(weight_col.as_ptr());
                if let Some(b) = bias_expanded.as_deref() {
                    out.push(b.as_ptr());
                }
            }
            _ => {}
        });
        out
    }

    pub(crate) fn lin(name: &str, input: &str) -> GraphNode {
        let w = arr2(&[[0.7_f32, -0.3], [0.2, 0.6]]);
        let b = arr1(&[0.05_f32, -0.04]);
        let layer = Layer::Linear(LinearLayer::new(w, Some(b)).expect("valid linear"));
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }

    /// The stacked-identity-blocks fixture from the resnet_decompose fold-order
    /// tests: input → l0 → block1(z=l0) → block2(z=add1) → lout.
    pub(crate) fn stacked_linear_fixture() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(lin("l0", NETWORK_INPUT));
        g.add_node(relu("relu1", "l0"));
        g.add_node(lin("l1a", "relu1"));
        g.add_node(add("add1", "l1a", "l0"));
        g.add_node(relu("relu2", "add1"));
        g.add_node(lin("l2a", "relu2"));
        g.add_node(add("add2", "l2a", "add1"));
        g.add_node(lin("lout", "add2"));
        g.set_output("lout");
        g
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::network::core::NETWORK_INPUT;

    fn count_dual_alpha(segments: &[GpuResnetSegment]) -> usize {
        let mut n = 0;
        for_each_layer(segments, |l| {
            if matches!(l, GpuCrownLayer::ActivationReluDualAlpha { .. }) {
                n += 1;
            }
        });
        n
    }

    fn layer_has_nan(layer: &GpuCrownLayer) -> bool {
        let any_nan = |v: &[f32]| v.iter().any(|x| x.is_nan());
        match layer {
            GpuCrownLayer::Linear { weight, bias, .. } => {
                any_nan(weight) || bias.as_deref().is_some_and(any_nan)
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                ..
            } => {
                any_nan(lower_slope)
                    || any_nan(upper_slope)
                    || any_nan(lower_intercept)
                    || any_nan(upper_intercept)
            }
            GpuCrownLayer::ActivationReluDualAlpha {
                lower_pos_slope,
                cross_slope,
                upper_neg_slope,
                cross_intercept,
                ..
            } => {
                any_nan(lower_pos_slope)
                    || any_nan(cross_slope)
                    || any_nan(upper_neg_slope)
                    || any_nan(cross_intercept)
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                ..
            } => any_nan(weight_col) || bias_expanded.as_deref().is_some_and(any_nan),
            GpuCrownLayer::MaxPool2d {
                ibp_lower,
                ibp_upper,
                ..
            } => any_nan(ibp_lower) || any_nan(ibp_upper),
        }
    }

    fn slot_layer<'s>(segments: &'s [GpuResnetSegment], slot: &DynamicSlot) -> &'s GpuCrownLayer {
        let layers = match (&segments[slot.seg_idx], slot.branch) {
            (GpuResnetSegment::Chain(v) | GpuResnetSegment::Residual(v), SlotBranch::Main) => v,
            (GpuResnetSegment::ResidualProj(f, _), SlotBranch::Main) => f,
            (GpuResnetSegment::ResidualProj(_, p), SlotBranch::Proj) => p,
            _ => panic!("slot does not fit segment structure"),
        };
        &layers[slot.layer_idx]
    }

    // ---------- oracles ----------

    /// THE increment's core oracle: build the skeleton from exemplar domain A
    /// (alpha present AND absent), fold for a DIFFERENT domain B (perturbed
    /// input box → perturbed bounds; its own alpha bridge, present AND absent),
    /// and assert the fold's `(segments, relu_names, frontier_abs, node_abs)`
    /// is BIT-identical to legacy `extract_gpu_segments_with_relu_names_ext`
    /// run directly on B. Also pins the fixture's segment/fold-order shape and
    /// the cross-fold static-`Arc` sharing.
    #[test]
    fn skeleton_fold_bit_identical_to_legacy_on_conv_resnet() {
        let graph = conv_resnet_fixture();
        let input_a = box_input(&[2, 6, 6], -1.0, 1.0);
        let input_b = box_input(&[2, 6, 6], -0.8, 0.9);
        let bounds_a = graph.collect_node_bounds(&input_a).expect("bounds A");
        let bounds_b = graph.collect_node_bounds(&input_b).expect("bounds B");
        let alpha_a = mk_alpha(&graph, &bounds_a, &CONV_FIXTURE_RELUS, 0.35, 0.65);
        let alpha_b = mk_alpha(&graph, &bounds_b, &CONV_FIXTURE_RELUS, 0.25, 0.75);

        let skel_alpha = build_resnet_segment_skeleton(
            &graph,
            &input_a,
            "conv_out",
            &bounds_a,
            &bounds_a,
            Some(&alpha_a),
            false,
        )
        .expect("skeleton builds (alpha exemplar)");
        let skel_plain = build_resnet_segment_skeleton(
            &graph, &input_a, "conv_out", &bounds_a, &bounds_a, None, false,
        )
        .expect("skeleton builds (no-alpha exemplar)");
        assert!(skel_alpha.matches_graph(&graph));
        assert_eq!(skel_alpha.cache_key(), ("conv_out", false));

        for (alpha, atag) in [(Some(&alpha_b), "alpha-B"), (None, "no-alpha")] {
            let legacy = legacy_ext(
                &graph, &input_b, "conv_out", &bounds_b, &bounds_b, alpha, false,
            )
            .expect("legacy extracts B");
            // Pin the fixture's structure (the fold-order contract, module doc §6).
            assert_eq!(legacy.0.len(), 5, "fixture: 5 segments");
            assert!(matches!(legacy.0[1], GpuResnetSegment::ResidualProj(_, _)));
            assert!(matches!(legacy.0[3], GpuResnetSegment::Residual(_)));
            assert_eq!(
                legacy.1,
                vec!["relu_out", "b2r1", "b1r1", "relu0"],
                "fixture: fold-order relu names"
            );
            for (skel, stag) in [(&skel_alpha, "alpha-built"), (&skel_plain, "plain-built")] {
                let fold = skel
                    .fold_for_domain(&graph, &input_b, &bounds_b, &bounds_b, alpha)
                    .expect("fold succeeds where legacy succeeds");
                assert_extraction_bits_eq(&fold, &legacy, &format!("{stag}/{atag}"));
            }
        }

        // Two folds of the same skeleton share every static Arc payload
        // (the cross-batch `arc_slice_eq`/`ptr_eq` payoff this split exists for).
        let f1 = skel_plain
            .fold_for_domain(&graph, &input_b, &bounds_b, &bounds_b, None)
            .expect("fold 1");
        let f2 = skel_plain
            .fold_for_domain(&graph, &input_b, &bounds_b, &bounds_b, Some(&alpha_b))
            .expect("fold 2");
        let p1 = static_arc_ptrs(&f1.0);
        assert!(!p1.is_empty(), "fixture has static Arc payloads");
        assert_eq!(
            p1,
            static_arc_ptrs(&f2.0),
            "static Arcs shared across folds"
        );
    }

    /// Subtlety: the ReLU layer VARIANT is per-domain — the same skeleton folds
    /// to `ActivationReluDualAlpha` under a dual-alpha bridge and to plain
    /// `Activation` with alpha absent. The variant is never frozen at build.
    #[test]
    fn skeleton_fold_decides_relu_variant_per_domain() {
        let graph = conv_resnet_fixture();
        let input_a = box_input(&[2, 6, 6], -1.0, 1.0);
        let input_b = box_input(&[2, 6, 6], -0.8, 0.9);
        let bounds_a = graph.collect_node_bounds(&input_a).expect("bounds A");
        let bounds_b = graph.collect_node_bounds(&input_b).expect("bounds B");
        let alpha_b = mk_alpha(&graph, &bounds_b, &CONV_FIXTURE_RELUS, 0.25, 0.75);

        let skel = build_resnet_segment_skeleton(
            &graph, &input_a, "conv_out", &bounds_a, &bounds_a, None, false,
        )
        .expect("skeleton builds");
        let fold_dual = skel
            .fold_for_domain(&graph, &input_b, &bounds_b, &bounds_b, Some(&alpha_b))
            .expect("fold with dual alpha");
        assert!(
            count_dual_alpha(&fold_dual.0) > 0,
            "dual-alpha domain must produce ActivationReluDualAlpha \
             (fixture needs an unstable neuron)"
        );
        let fold_plain = skel
            .fold_for_domain(&graph, &input_b, &bounds_b, &bounds_b, None)
            .expect("fold without alpha");
        assert_eq!(
            count_dual_alpha(&fold_plain.0),
            0,
            "no-alpha domain must stay on the symmetric Activation variant"
        );
    }

    /// `None`-agreement: whenever legacy extraction refuses a domain (missing
    /// bounds entry for a branch node, a frontier/chain node, the trailing
    /// chain), the fold must refuse the SAME domain. Also: the BUILD refuses
    /// when the exemplar itself is un-extractable.
    #[test]
    fn skeleton_fold_none_agreement_with_legacy_refusals() {
        let graph = conv_resnet_fixture();
        let input_a = box_input(&[2, 6, 6], -1.0, 1.0);
        let input_b = box_input(&[2, 6, 6], -0.8, 0.9);
        let bounds_a = graph.collect_node_bounds(&input_a).expect("bounds A");
        let bounds_b = graph.collect_node_bounds(&input_b).expect("bounds B");
        let skel = build_resnet_segment_skeleton(
            &graph, &input_a, "conv_out", &bounds_a, &bounds_a, None, false,
        )
        .expect("skeleton builds");

        // b1r1: F-branch interior; add1: chain input + chain frontier; relu0:
        // trailing-chain input; add2: chain input + frontier of the first chain.
        for missing in ["b1r1", "add1", "relu0", "add2"] {
            let mut broken = bounds_b.clone();
            assert!(broken.remove(missing).is_some(), "fixture node {missing}");
            let legacy = legacy_ext(&graph, &input_b, "conv_out", &broken, &broken, None, false);
            assert!(legacy.is_none(), "legacy must refuse without {missing}");
            let fold = skel.fold_for_domain(&graph, &input_b, &broken, &broken, None);
            assert!(
                fold.is_none(),
                "fold must refuse without {missing} (None-agreement)"
            );
        }

        // Build-side agreement: an un-extractable exemplar refuses the build.
        let mut broken_a = bounds_a;
        broken_a.remove("b1r1");
        assert!(
            build_resnet_segment_skeleton(
                &graph, &input_a, "conv_out", &broken_a, &broken_a, None, false,
            )
            .is_none(),
            "build must refuse when legacy would refuse the exemplar"
        );
    }

    /// Subtlety: dynamic slots are NaN-poisoned in the skeleton; static
    /// payloads — including the constant-arith `Activation` (classified by
    /// build-time ORIGIN, not variant) — are untouched and finite.
    #[test]
    fn skeleton_dynamic_slots_nan_poisoned_static_payloads_clean() {
        let graph = conv_resnet_fixture();
        let input_a = box_input(&[2, 6, 6], -1.0, 1.0);
        let bounds_a = graph.collect_node_bounds(&input_a).expect("bounds A");
        let alpha_a = mk_alpha(&graph, &bounds_a, &CONV_FIXTURE_RELUS, 0.35, 0.65);
        let skel = build_resnet_segment_skeleton(
            &graph,
            &input_a,
            "conv_out",
            &bounds_a,
            &bounds_a,
            Some(&alpha_a),
            false,
        )
        .expect("skeleton builds");

        let mut relu_slots = 0usize;
        let mut pool_slots = 0usize;
        for slot in &skel.dynamic_slots {
            match (&slot.kind, slot_layer(&skel.segments, slot)) {
                (
                    SlotKind::Relu { .. },
                    GpuCrownLayer::Activation {
                        lower_slope,
                        upper_slope,
                        lower_intercept,
                        upper_intercept,
                        ..
                    },
                ) => {
                    relu_slots += 1;
                    for v in [lower_slope, upper_slope, lower_intercept, upper_intercept] {
                        assert!(
                            v.iter().all(|x| x.is_nan()),
                            "ReLU slot placeholder must be all-NaN"
                        );
                    }
                }
                (
                    SlotKind::MaxPool { .. },
                    GpuCrownLayer::MaxPool2d {
                        routing,
                        ibp_lower,
                        ibp_upper,
                        ..
                    },
                ) => {
                    pool_slots += 1;
                    assert!(
                        routing.iter().all(|&r| r == u32::MAX),
                        "MaxPool slot placeholder must route nothing (all-IBP)"
                    );
                    assert!(
                        ibp_lower.iter().chain(ibp_upper.iter()).all(|x| x.is_nan()),
                        "MaxPool slot IBP windows must be all-NaN"
                    );
                }
                _ => panic!("slot kind / placeholder variant mismatch"),
            }
        }
        assert_eq!(relu_slots, skel.relu_names.len(), "one slot per fold ReLU");
        assert_eq!(relu_slots, 4, "fixture: 4 ReLU slots");
        assert_eq!(pool_slots, 1, "fixture: 1 MaxPool slot");

        // Every NON-slot layer is static and must carry no NaN — in particular
        // the AddConstant `Activation` (segment 2's only layer).
        let slot_positions: std::collections::HashSet<(usize, u8, usize)> = skel
            .dynamic_slots
            .iter()
            .map(|s| {
                (
                    s.seg_idx,
                    match s.branch {
                        SlotBranch::Main => 0u8,
                        SlotBranch::Proj => 1u8,
                    },
                    s.layer_idx,
                )
            })
            .collect();
        for (seg_idx, seg) in skel.segments.iter().enumerate() {
            let branches: [(&[GpuCrownLayer], u8); 2] = match seg {
                GpuResnetSegment::Chain(v) | GpuResnetSegment::Residual(v) => {
                    [(&v[..], 0), (&[], 1)]
                }
                GpuResnetSegment::ResidualProj(f, p) => [(&f[..], 0), (&p[..], 1)],
            };
            for (layers, btag) in branches {
                for (layer_idx, layer) in layers.iter().enumerate() {
                    if !slot_positions.contains(&(seg_idx, btag, layer_idx)) {
                        assert!(
                            !layer_has_nan(layer),
                            "static layer (seg {seg_idx} branch {btag} idx {layer_idx}) \
                             must not be poisoned"
                        );
                    }
                }
            }
        }
        assert!(
            matches!(
                &skel.segments[2],
                GpuResnetSegment::Chain(v) if v.len() == 1
                    && matches!(&v[0], GpuCrownLayer::Activation { .. })
                    && !layer_has_nan(&v[0])
            ),
            "constant-arith Activation is STATIC (origin-classified) and clean"
        );
    }

    /// Stale-skeleton guard: `matches_graph` accepts the (identically rebuilt)
    /// source graph and rejects a graph whose node sequence differs.
    #[test]
    fn skeleton_matches_graph_staleness_guard() {
        let graph = conv_resnet_fixture();
        let input_a = box_input(&[2, 6, 6], -1.0, 1.0);
        let bounds_a = graph.collect_node_bounds(&input_a).expect("bounds A");
        let skel = build_resnet_segment_skeleton(
            &graph, &input_a, "conv_out", &bounds_a, &bounds_a, None, false,
        )
        .expect("skeleton builds");

        assert!(skel.matches_graph(&graph));
        let rebuilt = conv_resnet_fixture();
        assert!(
            skel.matches_graph(&rebuilt),
            "identical construction order must match"
        );
        let mut extended = conv_resnet_fixture();
        extended.add_node(relu("extra", "conv_out"));
        assert!(
            !skel.matches_graph(&extended),
            "a graph with an extra node must be rejected"
        );
    }

    /// Byte-identity oracle on the 2-dim stacked linear resnet (the fold-order
    /// fixture), pinning `relu_names == [relu2, relu1]` through the fold.
    #[test]
    fn skeleton_fold_bit_identical_on_stacked_linear_resnet() {
        let graph = stacked_linear_fixture();
        let input_a = box_input(&[2], -0.5, 0.5);
        let input_b = box_input(&[2], -0.4, 0.45);
        let bounds_a = graph.collect_node_bounds(&input_a).expect("bounds A");
        let bounds_b = graph.collect_node_bounds(&input_b).expect("bounds B");
        let alpha_b = mk_alpha(&graph, &bounds_b, &["relu1", "relu2"], 0.2, 0.8);

        let skel = build_resnet_segment_skeleton(
            &graph, &input_a, "lout", &bounds_a, &bounds_a, None, false,
        )
        .expect("skeleton builds");
        for (alpha, atag) in [(Some(&alpha_b), "alpha-B"), (None, "no-alpha")] {
            let legacy = legacy_ext(&graph, &input_b, "lout", &bounds_b, &bounds_b, alpha, false)
                .expect("legacy extracts B");
            assert_eq!(legacy.1, vec!["relu2", "relu1"], "fold-order names");
            let fold = skel
                .fold_for_domain(&graph, &input_b, &bounds_b, &bounds_b, alpha)
                .expect("fold succeeds");
            assert_extraction_bits_eq(&fold, &legacy, atag);
        }
    }

    /// #metaroom-chain-wide agreement: with `allow_pure_chain = true` a pure
    /// chain builds + folds bit-identically; with `false` BOTH the build and
    /// legacy refuse (None-agreement at the >=1-residual gate).
    #[test]
    fn skeleton_pure_chain_gate_agreement() {
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2", "relu1"));
        g.set_output("l2");
        let input_a = box_input(&[2], -0.5, 0.5);
        let input_b = box_input(&[2], -0.3, 0.6);
        let bounds_a = g.collect_node_bounds(&input_a).expect("bounds A");
        let bounds_b = g.collect_node_bounds(&input_b).expect("bounds B");

        let skel =
            build_resnet_segment_skeleton(&g, &input_a, "l2", &bounds_a, &bounds_a, None, true)
                .expect("pure chain builds with allow_pure_chain");
        assert_eq!(skel.cache_key(), ("l2", true));
        let legacy = legacy_ext(&g, &input_b, "l2", &bounds_b, &bounds_b, None, true)
            .expect("legacy extracts pure chain");
        let fold = skel
            .fold_for_domain(&g, &input_b, &bounds_b, &bounds_b, None)
            .expect("fold succeeds");
        assert_extraction_bits_eq(&fold, &legacy, "pure-chain");

        assert!(
            build_resnet_segment_skeleton(&g, &input_a, "l2", &bounds_a, &bounds_a, None, false)
                .is_none(),
            "allow_pure_chain=false keeps the >=1-residual refusal at build"
        );
        assert!(
            legacy_ext(&g, &input_b, "l2", &bounds_b, &bounds_b, None, false).is_none(),
            "legacy agrees (None-agreement at the gate)"
        );
    }
}
