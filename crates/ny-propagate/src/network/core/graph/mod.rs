// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph (DAG-based) network representation and propagation.
//!
//! This module provides `GraphNetwork`, a neural network representation as a
//! directed acyclic graph (DAG). Unlike the sequential `Network`, `GraphNetwork`
//! can represent branching computations like attention (Q/K/V projections,
//! matmul, softmax).
//!
//! # Example: Simplified Attention
//! ```text
//!              input
//!             /  |  \
//!          Q_proj K_proj V_proj
//!             \  |      |
//!              \ |      |
//!           matmul(Q,K^T)
//!                |      |
//!             softmax   |
//!                \     /
//!              matmul(attn, V)
//!                   |
//!                output
//! ```
//!
//! # Module Structure
//!
//! - `node.rs` - GraphNode struct definition
//! - `config.rs` - Layer configuration methods (LayerNorm, GELU, etc.)
//! - `traversal.rs` - Topological sort algorithm
//! - `crown.rs` - CROWN propagation methods
//! - `crown_batched.rs` - N-D batched CROWN propagation
//! - `convert.rs` - Conversion from sequential Network
//! - `forward_linear.rs` - Forward-linear intermediate bound collection for DAGs
//! - `ibp/` - IBP propagation (existing submodule)

mod alpha_crown_batched;
pub(crate) mod backward_helpers;
pub(crate) mod batched_accumulator;
mod config;
mod convert;
mod crown;
mod crown_batched;
pub mod crown_block_wise;
pub(crate) mod dispatch_plan;
mod fallback_reason;
mod forward_linear;
pub(crate) mod ibp;
mod maxpool_relu;
pub(crate) mod merge_accumulator;
mod node;
mod shape_contract;
mod softmax_complex;
mod traversal;

use std::{collections::HashMap, sync::OnceLock};

use ny_core::{NyError, Result};
use tracing::debug;

use crate::layers::attention::{AttentionMask, SelfAttentionLayer};
use crate::layers::binary_ops::BilinearCrownLayer;
use crate::layers::softmax::{CausalSoftmaxLayer, SoftmaxLayer};
use crate::layers::Layer;

pub(crate) use backward_helpers::{
    apply_dense_backward_dispatch_result, try_dense_spatial_patches_reentry,
};
pub(crate) use fallback_reason::graph_crown_dispatch_fallback_reason;
pub use ibp::{ZonotopePropagationOptions, ZonotopeSoftmaxMode};
pub use maxpool_relu::{VggMaxPoolRewriteMode, VggMaxPoolRewriteReport};
pub use node::{GraphNode, NETWORK_INPUT};
pub(crate) use shape_contract::GraphTargetShapeContract;
pub use softmax_complex::{SoftmaxComplexReport, SOFTMAX_COMPLEX_SHIFT_GUARD};

/// Input-keyed single-entry cache for the certified forward-linear reference
/// map. The SAME root input is bound 3+ times per instance (PGD spec-CROWN
/// prechecks, alpha reference collection, spec-propagation setup) at ~22s per
/// computation on cifar100-scale conv DAGs; this collapses that to one.
/// Per-domain BaB inputs differ bit-wise and simply miss.
///
/// The entry also retains the OUTPUT node's certified forward-linear
/// `LinearBounds` (#w4-root-margin) when the pass produced one — the affine
/// map w.r.t. the original input that the C-matrix margin composition needs
/// (output-dim × input-dim; ~2.5 MB on cifar100, freed with the entry). `None`
/// when the pass did not retain the output map.
///
/// Clone RESETS the cache: a cloned graph may be mutated (patches mode, sound
/// modes, node edits) in ways that change bounds, so sharing cached bounds
/// across clones would be a staleness hazard. Semantic `&mut self` mutators
/// must call [`GraphNetwork::invalidate_forward_linear_cache`].
///
/// Two independent single-entry slots (#w4-root-alpha): `fixed` holds the
/// adaptive-slope map (key = input-bits hash) and `alpha` holds the
/// alpha-fed map (key = input-bits hash combined with a fingerprint of the
/// per-node alpha vectors). Separate slots prevent the two passes from
/// thrashing each other's entry when both run on the same root input.
#[derive(Debug, Default)]
pub(crate) struct ForwardLinearMapCache {
    pub(crate) fixed: std::sync::RwLock<Option<ForwardLinearCacheEntry>>,
    pub(crate) alpha: std::sync::RwLock<Option<ForwardLinearCacheEntry>>,
    /// Memo for the forward-map alpha OPTIMIZER (#w4-root-alpha-opt), keyed by
    /// (input bits, spec-matrix bits): the optimized per-node alphas plus the
    /// run's stats, or `None` when the optimizer declined (no straggler rows /
    /// no predicted improvement). Prevents re-paying the sweeps when the same
    /// root spec request is re-bounded.
    #[allow(clippy::type_complexity)]
    pub(crate) alpha_opt: std::sync::RwLock<
        Option<(
            u64,
            Option<(
                std::sync::Arc<std::collections::BTreeMap<String, ndarray::Array1<f32>>>,
                forward_linear::alpha_opt::AlphaOptStats,
            )>,
        )>,
    >,
}

/// One cached forward-linear result: (key, concretized node-bounds map,
/// retained OUTPUT-node `LinearBounds` when produced, FRESH build duration).
/// The duration is the measured wall cost of the O(L) certified pass that
/// produced the entry — the alpha-fed rebuild consults it to decide whether
/// a second pass fits the remaining deadline (#w4-root-alpha) instead of
/// burning BaB budget on a doomed attempt.
pub(crate) type ForwardLinearCacheEntry = (
    u64,
    std::sync::Arc<HashMap<String, ny_tensor::BoundedTensor>>,
    Option<std::sync::Arc<crate::bounds::LinearBounds>>,
    std::time::Duration,
);

impl Clone for ForwardLinearMapCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

/// Input-keyed cache for the FULL graph CROWN-IBP collection
/// (#cgan-collection-cache).
///
/// # Soundness invariant
/// A CROWN/IBP node-bounds map computed for input box `B` on network `N` is a
/// set of valid enclosures for `(N, B)` FOREVER — reuse is sound iff the
/// network is the same object/weights and the input box is BIT-EXACT identical
/// (the f32 lower/upper bit patterns, hashed by the cache key). It must NEVER
/// be reused across different boxes: BaB children split the box, so their
/// bit-different inputs hash to different keys and miss. (Passing the ROOT
/// map as *reference bounds* to child domains remains the existing, separate,
/// intentionally-sound mechanism — reference bounds are re-intersected per
/// child, not substituted.)
///
/// # Why
/// On cgan_2023 the pipeline computed a COMPLETE CROWN-IBP collection at the
/// disjunctive precheck (~71 s, all 8 pre-ReLU nodes) and then RE-RAN the same
/// collection up to 5 more times (alpha-warmup reference bounds, iteration-0
/// output backward, sequential-alpha re-reference), each re-run truncated by
/// the remaining phase budget — and the LAST truncated (mostly-IBP) map was
/// what reached BaB. This cache makes every later same-box consumer reuse the
/// first complete map.
///
/// # Replacement policy
/// A bounded LRU of per-key entries (`CROWN_IBP_COLLECTION_CACHE_CAP`,
/// #cgan-collection-multislot). Within a key the store MERGES same-box maps by
/// per-element intersection (both are valid enclosures of the identical box, so
/// the intersection is a valid enclosure at least as tight as either) and keeps
/// the most-complete result; a truncated collection is never served as complete
/// (see `crown_ibp_collection_store` / `_lookup`). Across keys, entries never
/// interact — each is served only on an exact key+fingerprint match — so a
/// different-key store can no longer evict a complete map another consumer will
/// reuse, and LRU eviction only ever drops a least-recent DIFFERENT-box entry.
///
/// Clone RESETS the cache (same rationale as [`ForwardLinearMapCache`]: a
/// cloned graph may be mutated into a semantically different network).
/// Pure-clone call sites adopt via
/// [`GraphNetwork::adopt_crown_ibp_collection_cache_from`]. Every `&mut self`
/// mutator invalidates via [`GraphNetwork::invalidate_forward_linear_cache`],
/// which clears this cache too.
/// Max distinct-key entries retained (#cgan-collection-multislot). Small: the
/// root box plus a few in-flight child boxes / coverage-descriptor variants.
///
/// A SINGLE slot let a different-key store (a BaB child box, or a consumer whose
/// coverage descriptor — engine presence / conv-mode — differs) EVICT a COMPLETE
/// root-box map, so a later same-box consumer missed the complete entry and
/// recomputed a budget-truncated one ("the last, most-truncated map reached BaB
/// instead of the complete one"). Retaining several bit-exact-keyed entries
/// removes that cross-key eviction. Each entry is an INDEPENDENT valid enclosure
/// for its own input box (bit-exact key + fingerprint); entries never interact
/// and are served only on an exact key+fingerprint match, so this can never
/// serve a wrong or looser bound.
pub(crate) const CROWN_IBP_COLLECTION_CACHE_CAP: usize = 8;

#[derive(Debug, Default)]
pub(crate) struct CrownIbpCollectionCache {
    /// Bounded LRU of per-key entries, most-recently-stored first. Replaces the
    /// former single slot (#cgan-collection-multislot) so a different-key store
    /// no longer evicts a complete map another consumer will reuse.
    pub(crate) slots: std::sync::RwLock<Vec<CrownIbpCollectionCacheEntry>>,
    /// Number of times a collection was served from this cache (diagnostics
    /// and the integration-test hook proving the backward ran once).
    pub(crate) hits: std::sync::atomic::AtomicUsize,
}

/// One cached CROWN-IBP collection result.
#[derive(Debug, Clone)]
pub(crate) struct CrownIbpCollectionCacheEntry {
    /// Bit-exact input-box key (plus collection-coverage descriptor); see
    /// `GraphNetwork::crown_ibp_collection_cache_key`. This is a HASH — the
    /// `fingerprint` below is the collision-proof ground truth.
    pub(crate) key: u64,
    /// The EXACT byte string the key hashes (input-box f32 bits + shape +
    /// coverage descriptor). Compared on every hit and merge so a u64 hash
    /// collision across different boxes can never serve or merge a wrong map
    /// (adversarial-review hardening, #cgan-collection-cache).
    pub(crate) fingerprint: std::sync::Arc<[u8]>,
    pub(crate) result: std::sync::Arc<crate::types::GraphCrownIbpBoundsResult>,
    /// No time-budget truncation occurred (no Deadline/PerNodeDeadline/
    /// PatchesBudget fallback events): re-running could never compute more.
    pub(crate) complete: bool,
    /// Number of nodes with `BoundsProvenance::Crown` — the completeness
    /// order used when comparing two truncated collections.
    pub(crate) crown_count: usize,
}

impl Clone for CrownIbpCollectionCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

/// A neural network represented as a directed acyclic graph (DAG).
///
/// Unlike `Network` which is sequential, `GraphNetwork` can represent
/// branching computations like attention (Q/K/V projections, matmul, softmax).
#[derive(Debug, Clone)]
pub struct GraphNetwork {
    /// All nodes in the graph, keyed by name.
    pub(crate) nodes: HashMap<String, GraphNode>,
    /// Order of node names for iteration.
    pub(crate) node_order: Vec<String>,
    /// Name of the output node.
    pub(crate) output_node: String,
    /// Whether to use Patches mode for Conv2d CROWN backward.
    /// When `false`, all CROWN backward paths use Dense (Matrix) mode.
    /// Set from `BetaCrownConfig::use_patches()` by the verifier.
    /// Default: `true` (existing behavior — Patches for spatial Conv2d).
    /// Reference: alpha-beta-CROWN `general.conv_mode` (`abcrown.py:228-231`).
    pub(crate) use_patches_mode: bool,
    /// Per-node CROWN-IBP time-budget policy overrides (#4413, #cgan-bn11-budget).
    /// Set from `BetaCrownConfig::crown_ibp_per_node_time_budget()` by the
    /// verifier (same plumbing pattern as `use_patches_mode`); carried by
    /// `Clone` so every configured graph copy inherits it. Default: all-`None`
    /// (the built-in 2 s floor / 12 s cap constants).
    pub(crate) crown_ibp_per_node_time_budget: crate::types::CrownIbpPerNodeTimeBudget,
    /// Cached topological execution order for hot CROWN backward paths.
    pub(crate) cached_exec_order: OnceLock<Vec<String>>,
    /// Cached pre-compiled dispatch plan for CROWN backward loops.
    /// Built lazily on first access via [`Self::dispatch_plan()`].
    pub(crate) cached_dispatch_plan: OnceLock<dispatch_plan::CrownDispatchPlan>,
    /// Cached per-node ancestor sets (nodes reachable backward from each target),
    /// returned in topological order. Eliminates O(N^2) redundant BFS in CROWN-IBP
    /// collection (#2220 Packet A, #2237 F1).
    pub(crate) cached_ancestors: OnceLock<HashMap<String, Vec<String>>>,
    /// See [`ForwardLinearMapCache`].
    pub(crate) cached_forward_linear_map: ForwardLinearMapCache,
    /// See [`CrownIbpCollectionCache`] (#cgan-collection-cache).
    pub(crate) cached_crown_ibp_collection: CrownIbpCollectionCache,
    /// Declared per-node output shapes from load-time shape inference
    /// (internal, unbatched convention), keyed by node name. Metadata only:
    /// used to shape the conservative `[-inf, +inf]` substitution when the
    /// taint-gated IBP degrade path recovers from a propagation error at a
    /// node downstream of an OpaqueSkip (cctsdb_yolo_2023). Never read for
    /// finite bound values.
    pub(crate) declared_shapes: HashMap<String, Vec<usize>>,
    /// Process-unique identity of this graph instance for the dark cut-fold
    /// registry (`NY_CUT_FOLD`, Certified Cut-CROWN C2). Minted per
    /// [`GraphNetwork::new`]; copied by `Clone` while the clone remains
    /// semantically identical, and refreshed by every structural mutation or
    /// output retarget. Keying registrations and exact bound caches by this
    /// token means data derived for one model can never reach a same-shaped but
    /// different graph in the same process — that would be unsound, not just
    /// noisy. See `bab_cuts::CutFoldScope`.
    pub(crate) cut_fold_scope: crate::beta_crown::bab_cuts::CutFoldScope,
}

/// Deep CNN DAGs use plain IBP intermediates for the final CROWN pass once the
/// graph grows past this threshold.
///
/// Reference: alpha-beta-CROWN maps `crown-ibp`/`ibp+crown` to `IBP=True`
/// plus a single backward pass, rather than tightening every intermediate node
/// with an O(N^2) loop. See `auto_LiRPA/bound_general.py:1283-1284,1480-1526`.
pub(crate) const CROWN_IBP_PER_NODE_THRESHOLD: usize = 50;

/// #binary-relax-crown-ibp (DARK): read once, so a mid-run env change cannot
/// make two collectors on the same graph disagree about the arm.
pub(crate) fn crown_ibp_binary_relax_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NY_CROWN_IBP_BINARY")
            .ok()
            .is_some_and(|v| v == "1")
    })
}

impl GraphNetwork {
    /// Decide whether to use the expensive O(N²) CROWN-IBP intermediate tightening pass.
    ///
    /// For CNN-style DAGs (e.g., ResNets), CROWN-IBP intermediates dramatically improve
    /// ReLU relaxations across skip connections.
    ///
    /// For transformer-style graphs with binary relaxation ops (MatMul/MulBinary), we
    /// conservatively prefer the forward IBP pass until demand-driven selection proves
    /// safe for those surfaces.
    ///
    /// Unary transformer ops (Softmax, CausalSoftmax, LayerNorm, RmsNorm, SiLU, GELU)
    /// are now ALLOWED (#3775): they already have direct CROWN backward support, and
    /// the demand-driven selection in `crown_tighten.rs` limits the O(N²) tightening
    /// loop to only the producer nodes that downstream nonlinear ops actually need.
    ///
    /// Still blocked: MatMul, MulBinary (binary relaxation — high runtime risk without
    /// per-input concreteness filtering), GroupNorm.
    ///
    /// InstanceNorm1d and AdaIN1d are NOT blocklisted: AdaIN now routes
    /// `IbpValidated` CROWN through its effective InstanceNorm1d equivalent, so
    /// both CNN-style normalization surfaces benefit from CROWN-IBP
    /// intermediate tightening. Unblocked by #3596 and #3912.
    pub(crate) fn should_use_crown_ibp_intermediates(&self) -> bool {
        // #binary-relax-crown-ibp (DARK, `NY_CROWN_IBP_BINARY=1`): the
        // binary-relaxation blocklist is a COST guard, not a soundness guard —
        // `crown_tighten` INTERSECTS every per-node CROWN bound with that
        // node's IBP bound (`intersection_per_element`, crown_tighten.rs:578),
        // so the collected map is elementwise never looser than the plain-IBP
        // map this refuses to. MatMul / MulBinary / BilinearCrown all have real
        // CROWN backward support (`dispatch_shared_core`), and the per-input
        // concreteness filter the #3775 comment asks for now exists in
        // `demand::nodes_requiring_crown_tightening` (it skips producers whose
        // IBP bounds are already concrete). The remaining exposure is runtime:
        // O(N²) backward passes on a graph that may carry many binary nodes.
        // Scoped to graphs at or under the per-node node-count threshold so the
        // dark lane can only touch small graphs. Gate-unset is byte-identical.
        //
        // MEASURED OUTCOME — the gate STAYS DARK. The full A/B promised by
        // 1f88d10d has now landed and REFUTES the lever end-to-end. The root
        // objective really is ~11x tighter and BaB's best_gap really does close
        // ~9x (-0.1726 -> -0.0184 on quadrotor2d_state_0), but no verdict moves:
        //
        //   lsnc_relu, all 80 instances, official 25s budget, preset loaded
        //     base     10 sat / 11 unknown / 59 timeout
        //     gate on  10 sat /  0 unknown / 70 timeout      0 conversions
        //   (an independent 20-instance INTERLEAVED base/gate-on rerun on the
        //    same box: 20/20 identical verdicts, 0 diffs)
        //   nn4sys mscn_128d + mscn_2048d (the only other in-scope models,
        //   47 nodes): 8/8 unsat -> 8/8 unsat, 1 timeout -> 1 timeout,
        //   mean wall 8.2s -> 9.1s.
        //
        // Diagnosis for the next lane: on lsnc the tightening lands on the
        // NON-binding rows. Row 0 of the disjunction goes [-540.27, 536.92] ->
        // [-48.40, 50.54], but the rows that actually gate the verdict
        // ([-30.087988, 30.087988] against thresholds 0.405468 / -0.355468) are
        // bit-identical in both arms. Tightening the reference map is therefore
        // NOT the lever for lsnc_relu; those rows need a different mechanism
        // (their width does not come from the intermediate relaxations this
        // collector repairs). Turning the gate on only spends ~22% of the
        // domain throughput for bounds the objective does not read.
        self.crown_ibp_intermediates_allowed(
            crown_ibp_binary_relax_enabled() && self.nodes.len() <= CROWN_IBP_PER_NODE_THRESHOLD,
        )
    }

    /// Pure predicate behind [`Self::should_use_crown_ibp_intermediates`].
    ///
    /// `binary_relax = false` is the shipped blocklist. `binary_relax = true`
    /// additionally admits the binary-relaxation ops (MatMul / BilinearCrown /
    /// MulBinary); `GroupNorm` stays blocked either way.
    pub(crate) fn crown_ibp_intermediates_allowed(&self, binary_relax: bool) -> bool {
        !self.nodes.values().any(|node| {
            if binary_relax {
                matches!(node.layer, Layer::GroupNorm(_))
            } else {
                matches!(
                    node.layer,
                    // Binary relaxation ops: conservative gate until
                    // demand-driven selection handles the per-input
                    // concreteness check (#3775).
                    Layer::MatMul(_)
                        | Layer::BilinearCrown(_)
                        | Layer::MulBinary(_)
                        // Not part of the immediate transformer unblock.
                        | Layer::GroupNorm(_)
                )
            }
        })
    }

    /// Whether this graph is small enough to justify per-node CROWN-IBP
    /// intermediate tightening instead of the reference-style IBP + final CROWN
    /// pass used for deep CNN DAGs.
    pub(crate) fn should_collect_per_node_crown_ibp_intermediates(&self) -> bool {
        self.should_use_crown_ibp_intermediates()
            && self.nodes.len() <= CROWN_IBP_PER_NODE_THRESHOLD
    }

    /// Whether this graph contains convolution layers (Conv/ConvTranspose, 1d/2d).
    pub(crate) fn has_conv_layers(&self) -> bool {
        self.nodes.values().any(|node| {
            matches!(
                node.layer,
                Layer::Conv2d(_)
                    | Layer::Conv1d(_)
                    | Layer::ConvTranspose2d(_)
                    | Layer::ConvTranspose1d(_)
            )
        })
    }

    /// Whether this graph contains 2-D convolutions — the image-scale surface
    /// where the certified forward-linear IMAGE pass (`forward_linear::image`)
    /// is the trigger and the whole-graph CPU spec-CROWN backward is
    /// deadline-infeasible (#w4-root-margin). Conv1d-only DAGs (nn4sys
    /// pensieve) are NOT image-scale: their spec-CROWN backward is cheap and
    /// measured tighter than the forward-linear C-margin composition
    /// (frac-head audit, 4726b45b).
    pub(crate) fn has_conv2d_layers(&self) -> bool {
        self.nodes
            .values()
            .any(|node| matches!(node.layer, Layer::Conv2d(_)))
    }

    /// Whether the graph contains a ConvTranspose2d node (#cgan-fwdlin-ref:
    /// scopes the dark sequential-chain image unlock to graphs that actually
    /// carry the certified ConvTranspose surface).
    pub(crate) fn has_conv_transpose2d_layers(&self) -> bool {
        self.nodes
            .values()
            .any(|node| matches!(node.layer, Layer::ConvTranspose2d(_)))
    }

    /// Whether it is sound to stack independent input domains along a fresh leading
    /// axis and run a single batched IBP forward (the input-split prescreen
    /// optimization, see `input_split::ibp_prescreen`).
    ///
    /// The batched prescreen prepends an `N` batch axis to the network input and
    /// relies on the forward pass treating that axis as a passive per-row dimension.
    /// That assumption holds for element-wise / last-axis ops (Gemm, ReLU, Add, …)
    /// but BREAKS for operators that reference an explicit ABSOLUTE axis: after the
    /// model is converted in the squeezed, unbatched convention, those axes index
    /// the feature dimension directly, so a prepended batch axis shifts every
    /// absolute axis by one. For example a `Gather(axis=0)` over a `[6]` feature
    /// vector becomes a gather over the `[N, 6]` batch axis once stacked — selecting
    /// across DOMAINS instead of features. This either errors out (index out of
    /// bounds) or, worse, silently mixes bounds from different domains and can mark a
    /// domain "verified" when it is not — an unsound verdict.
    ///
    /// When this returns `false`, the batched prescreen must be skipped; children
    /// then fall through to the per-domain bounding path (which runs each domain
    /// unbatched in its native shape and is unaffected). The prescreen is a pure
    /// speed enhancement, so skipping it never changes a verdict.
    ///
    /// This is a conservative deny-list of absolute-axis operators. It deliberately
    /// does NOT flag Flatten/Reshape/Squeeze/Unsqueeze (handled batch-transparently
    /// by the existing leading-axis-preserving reshape path) so the 100%-solve
    /// Conv/Gemm benchmarks (acasxu, metaroom, collins_rul_cnn, malbeware) keep the
    /// prescreen.
    pub(crate) fn is_input_split_batch_stack_safe(&self) -> bool {
        !self.nodes.values().any(|node| {
            matches!(
                node.layer,
                // Index / select / scatter along an absolute axis.
                Layer::Gather(_)
                    | Layer::ScatterNd(_)
                    | Layer::ScatterAdd(_)
                    | Layer::IndexAdd(_)
                    | Layer::Slice(_)
                    | Layer::Tile(_)
                    // Join / split along an absolute axis.
                    | Layer::Concat(_)
                    // Reductions over an absolute axis.
                    | Layer::ReduceSum(_)
                    | Layer::ReduceMean(_)
                    | Layer::ReduceMax(_)
                    | Layer::ReduceMin(_)
                    | Layer::CumSum(_)
                    | Layer::ArgMax(_)
                    | Layer::ArgMin(_)
                    | Layer::ArgSort(_)
                    | Layer::Topk(_)
                    // Axis-aware normalization / attention / softmax.
                    | Layer::Softmax(_)
                    | Layer::CausalSoftmax(_)
                    | Layer::LogSoftmax(_)
                    | Layer::LogSumExp(_)
                    | Layer::SelfAttention(_)
                    // BatchNorm's rank-3 layout is resolved by VALUE
                    // ([C, H, W] vs [N, C, L] decided by `shape[0] ==
                    // num_channels`), so stacking rank-2 feature maps prepends
                    // a batch axis that is mistaken for the channel axis
                    // whenever the child-domain count equals the channel count
                    // — every element of domain n is then silently scaled by
                    // channel n's affine. The sibling normalization layers
                    // (LayerNorm/RmsNorm/GroupNorm/InstanceNorm1d/AdaIN1d)
                    // resolve their axes relative to the TRAILING dimension,
                    // which is invariant under a prepended batch axis, so they
                    // stay off this list.
                    | Layer::BatchNorm(_)
                    // Axis permutation / padding / resize.
                    | Layer::Transpose(_)
                    | Layer::Pad(_)
                    | Layer::Resize(_)
            )
        })
    }

    /// Create a new empty graph network.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            node_order: Vec::new(),
            output_node: String::new(),
            use_patches_mode: true,
            crown_ibp_per_node_time_budget: crate::types::CrownIbpPerNodeTimeBudget::default(),
            cached_exec_order: OnceLock::new(),
            cached_dispatch_plan: OnceLock::new(),
            cached_ancestors: OnceLock::new(),
            cached_forward_linear_map: ForwardLinearMapCache::default(),
            cached_crown_ibp_collection: CrownIbpCollectionCache::default(),
            declared_shapes: HashMap::new(),
            cut_fold_scope: crate::beta_crown::bab_cuts::CutFoldScope::fresh(),
        }
    }

    /// This graph instance's identity token for the dark cut-fold registry
    /// (`NY_CUT_FOLD`, Certified Cut-CROWN C2). Pass it to
    /// `bab_cuts::set_cut_fold` when registering cuts derived FOR THIS graph;
    /// the fold site only applies entries registered under this token.
    pub fn cut_fold_scope(&self) -> crate::beta_crown::bab_cuts::CutFoldScope {
        self.cut_fold_scope
    }

    /// Drop the cached forward-linear reference map AND the cached CROWN-IBP
    /// collection (#cgan-collection-cache). MUST be called by every
    /// `&mut self` mutation that can change propagated bounds (node edits,
    /// patches mode, sound-mode flips) — a stale map is a soundness hazard.
    pub(crate) fn invalidate_forward_linear_cache(&mut self) {
        *self
            .cached_forward_linear_map
            .fixed
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        *self
            .cached_forward_linear_map
            .alpha
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        self.cached_crown_ibp_collection
            .slots
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    /// Adopt `source`'s cached forward-linear reference map (#w5-bab-throughput).
    ///
    /// `Clone` deliberately RESETS the cache (a clone may be mutated into a
    /// semantically different graph), which made every `configured_graph_for_crown`
    /// clone repay the full O(L) certified forward pass (~25s on cifar100) for a
    /// map already computed on the source. Callers that clone WITHOUT changing
    /// anything the forward-linear pass reads may adopt the source's cache.
    ///
    /// # Soundness
    /// The map is a pure function of (graph structure + weights, input-bits key).
    /// It is only valid to call this when `self` and `source` are structurally
    /// identical up to state the forward-linear collection never reads
    /// (`use_patches_mode` is CROWN-backward-only). Any later `&mut self`
    /// mutation still invalidates via `invalidate_forward_linear_cache`.
    pub(crate) fn adopt_forward_linear_cache_from(&mut self, source: &GraphNetwork) {
        let fixed = source
            .cached_forward_linear_map
            .fixed
            .read()
            .ok()
            .and_then(|guard| guard.clone());
        if fixed.is_some() {
            *self
                .cached_forward_linear_map
                .fixed
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = fixed;
        }
        let alpha = source
            .cached_forward_linear_map
            .alpha
            .read()
            .ok()
            .and_then(|guard| guard.clone());
        if alpha.is_some() {
            *self
                .cached_forward_linear_map
                .alpha
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = alpha;
        }
    }

    /// Adopt `source`'s cached CROWN-IBP collection (#cgan-collection-cache).
    ///
    /// # Soundness
    /// The cached map is a set of valid enclosures for (network weights,
    /// bit-exact input box). Only call this when `self` is a pure clone of
    /// `source` — same nodes and weights. Configuration knobs that change
    /// WHICH backward the collector runs (`use_patches_mode`) are part of the
    /// cache key, so a knob flip after adoption makes the entry miss instead
    /// of serving a map computed under a different policy; knobs that only
    /// change how much gets tightened (per-node time budget) do not affect
    /// entry validity — every entry remains a sound enclosure regardless.
    /// Any later `&mut self` mutation still invalidates via
    /// [`Self::invalidate_forward_linear_cache`].
    pub(crate) fn adopt_crown_ibp_collection_cache_from(&mut self, source: &GraphNetwork) {
        // Carry EVERY per-key entry (#cgan-collection-multislot): the precheck's
        // complete root-box map plus any other retained boxes. Each is a valid
        // enclosure for its own bit-exact box; adoption is a pure copy.
        let entries = source
            .cached_crown_ibp_collection
            .slots
            .read()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        if !entries.is_empty() {
            *self
                .cached_crown_ibp_collection
                .slots
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = entries;
        }
    }

    /// Adopt every input-keyed bound cache from `source` after a PURE clone
    /// (#cgan-collection-cache): the forward-linear reference map and the
    /// CROWN-IBP collection. `Clone` deliberately resets both; call sites
    /// that clone WITHOUT mutating anything the passes read (e.g. the
    /// disjunctive verify flow handing the precheck's graph to the BaB lane)
    /// use this so a map computed in an earlier phase is not recomputed under
    /// a smaller phase budget. See the per-cache adopt methods for the
    /// soundness contracts.
    pub fn adopt_bound_caches_from(&mut self, source: &GraphNetwork) {
        self.adopt_forward_linear_cache_from(source);
        self.adopt_crown_ibp_collection_cache_from(source);
    }

    /// Number of CROWN-IBP collections served from the input-keyed cache on
    /// this graph object (#cgan-collection-cache). Test hook + diagnostics.
    pub fn crown_ibp_collection_cache_hits(&self) -> usize {
        self.cached_crown_ibp_collection
            .hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record the declared (load-time shape-inferred) output shape for a node.
    ///
    /// Metadata only; used by the taint-gated IBP degrade path to shape
    /// conservative `[-inf, +inf]` substitutions. Does not affect any cached
    /// execution/dispatch state.
    pub fn set_declared_shape(&mut self, node: impl Into<String>, shape: Vec<usize>) {
        self.declared_shapes.insert(node.into(), shape);
    }

    /// Declared output shape for a node, if recorded at build time.
    pub fn declared_shape(&self, node: &str) -> Option<&[usize]> {
        self.declared_shapes.get(node).map(Vec::as_slice)
    }

    /// Output retargets only invalidate the plan's output-sensitive metadata.
    fn invalidate_dispatch_plan_cache(&mut self) {
        let _ = self.cached_dispatch_plan.take();
    }

    /// Structural graph mutations must clear all cached dispatch metadata and
    /// mint a new semantic identity. Config-only clones deliberately retain the
    /// scope; after any node rewrite they must no longer match source-model cuts
    /// or exact bound caches.
    fn invalidate_exec_order_cache(&mut self) {
        self.cut_fold_scope = crate::beta_crown::bab_cuts::CutFoldScope::fresh();
        let _ = self.cached_exec_order.take();
        let _ = self.cached_ancestors.take();
        self.invalidate_dispatch_plan_cache();
    }

    /// Get the pre-compiled dispatch plan, building it on first access.
    ///
    /// The plan caches name↔index mappings, per-node dispatch routes, and
    /// graph-level properties. All CROWN backward loops should use this
    /// instead of recomputing dispatch metadata per call.
    pub(crate) fn dispatch_plan(&self) -> Result<&dispatch_plan::CrownDispatchPlan> {
        if let Some(plan) = self.cached_dispatch_plan.get() {
            return Ok(plan);
        }
        let plan = dispatch_plan::CrownDispatchPlan::build(self)?;
        let _ = self.cached_dispatch_plan.set(plan);
        self.cached_dispatch_plan.get().ok_or_else(|| {
            NyError::InternalError("dispatch_plan cache missing after initialization".to_string())
        })
    }

    /// Set the conv mode policy from BetaCrownConfig.
    /// When `use_patches` is `false`, CROWN backward uses Dense (Matrix) mode
    /// instead of Patches for spatial Conv2d graphs.
    pub fn set_use_patches_mode(&mut self, use_patches: bool) {
        self.invalidate_forward_linear_cache();
        self.use_patches_mode = use_patches;
    }

    /// Set the per-node CROWN-IBP time-budget policy from BetaCrownConfig
    /// (#4413, #cgan-bn11-budget). Purely a time-vs-tightness policy: any
    /// value is sound (a skipped node degrades to IBP; a longer budget only
    /// lets the CROWN backward finish), so no bound cache is invalidated.
    pub fn set_crown_ibp_per_node_time_budget(
        &mut self,
        budget: crate::types::CrownIbpPerNodeTimeBudget,
    ) {
        self.crown_ibp_per_node_time_budget = budget;
    }

    /// Add a node to the graph.
    ///
    /// Nodes should be added in topological order (dependencies before dependents).
    ///
    /// # Panics
    /// Panics if a node with the same name already exists.
    ///
    /// Prefer [`try_add_node`](Self::try_add_node) in production code for explicit
    /// error handling. This method is retained for test convenience.
    pub fn add_node(&mut self, node: GraphNode) {
        self.invalidate_forward_linear_cache();
        let name = node.name.clone();
        assert!(
            !self.nodes.contains_key(&name),
            "Duplicate node '{name}' (#2136)"
        );
        self.invalidate_exec_order_cache();
        self.node_order.push(name.clone());
        self.nodes.insert(name, node);
    }

    /// Try to add a node to the graph, returning an error if the name already exists.
    ///
    /// This is the preferred method for production code — returns `Err` on duplicate
    /// names instead of panicking. All production graph-building code should use this.
    ///
    /// `SelfAttention` nodes are automatically decomposed into sub-nodes
    /// (BilinearCrown + Softmax + BilinearCrown) so each sub-layer's existing
    /// CROWN backward implementation is used. See [`Self::decompose_self_attention()`].
    /// This matches alpha-beta-CROWN's approach of handling attention via primitive ops.
    /// Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/`
    pub fn try_add_node(&mut self, node: GraphNode) -> Result<()> {
        self.invalidate_forward_linear_cache();
        if let Layer::SelfAttention(ref attn) = node.layer {
            return self.decompose_self_attention(&node.name, attn, &node.inputs);
        }
        // NOTE: Binary Div is NOT decomposed here. Unlike SelfAttention (which
        // decomposes into same-shape primitives), Div(a, b) typically involves
        // broadcasting (e.g., [1024] / scalar in L2 normalization) and MulBinary
        // CROWN backward doesn't support broadcasting. Instead, the spec-guided
        // CROWN backward handles Div via per-node IBP concretization (see
        // spec_propagation.rs BackwardDispatchResult::Unsupported). #3596
        let name = node.name.clone();
        if self.nodes.contains_key(&name) {
            return Err(NyError::InvalidSpec(format!(
                "Node '{}' already exists in graph",
                name
            )));
        }
        self.invalidate_exec_order_cache();
        self.node_order.push(name.clone());
        self.nodes.insert(name, node);
        Ok(())
    }

    /// Decompose a SelfAttention node into three primitive sub-nodes.
    ///
    /// Transforms: `SelfAttention(Q, K, V)` into a subgraph:
    ///   1. `{name}/qk`      = BilinearCrown(Q, K, transpose_b=true, scale)
    ///   2. `{name}/softmax`  = Softmax or CausalSoftmax on `{name}/qk`
    ///   3. `{name}`          = BilinearCrown(`{name}/softmax`, V, transpose_b=false)
    ///
    /// The final node keeps the original name so downstream references are valid.
    /// Requires `scale` to be explicitly set (not None) since we cannot infer
    /// head_dim from input shape at graph construction time.
    ///
    /// Reference: alpha-beta-CROWN decomposes attention at the ONNX graph level
    /// into MatMul + Softmax + MatMul primitives, each with `bound_backward`.
    fn decompose_self_attention(
        &mut self,
        name: &str,
        attn: &SelfAttentionLayer,
        inputs: &[String],
    ) -> Result<()> {
        if inputs.len() != 3 {
            return Err(NyError::InvalidSpec(format!(
                "SelfAttention '{}' requires exactly 3 inputs (Q, K, V) but got {}",
                name,
                inputs.len()
            )));
        }
        let scale = attn.scale.ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "SelfAttention '{}' requires explicit scale for graph decomposition \
                 (cannot infer head_dim at graph construction time)",
                name,
            ))
        })?;

        let q_input = &inputs[0];
        let k_input = &inputs[1];
        let v_input = &inputs[2];

        // Node 1: Q @ K^T (scaled) via BilinearCrown
        let qk_name = format!("{name}/qk");
        let qk_layer = Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(scale)));
        let qk_node = GraphNode::new(
            qk_name.clone(),
            qk_layer,
            vec![q_input.clone(), k_input.clone()],
        );

        // Node 2: Softmax (standard or causal)
        let softmax_name = format!("{name}/softmax");
        let softmax_layer = match attn.mask {
            AttentionMask::Standard => Layer::Softmax(SoftmaxLayer::new(-1)),
            AttentionMask::Causal => {
                let layer = match attn.window_size {
                    Some(window_size) => CausalSoftmaxLayer::new(-1).with_window_size(window_size),
                    None => CausalSoftmaxLayer::new(-1),
                };
                Layer::CausalSoftmax(layer)
            }
        };
        let softmax_node =
            GraphNode::new(softmax_name.clone(), softmax_layer, vec![qk_name.clone()]);

        // Node 3: probs @ V via BilinearCrown (keeps original name for downstream refs)
        let out_layer = Layer::BilinearCrown(BilinearCrownLayer::new(false, None));
        let out_node = GraphNode::new(
            name.to_string(),
            out_layer,
            vec![softmax_name.clone(), v_input.clone()],
        );

        // Validate the decomposed names before mutating the graph so cache
        // invalidation only happens when insertion will succeed.
        for node_name in [&qk_name, &softmax_name, &out_node.name] {
            if self.nodes.contains_key(node_name) {
                return Err(NyError::InvalidSpec(format!(
                    "Node '{}' already exists in graph \
                     (from SelfAttention decomposition of '{}')",
                    node_name, name
                )));
            }
        }

        self.invalidate_exec_order_cache();

        for node in [qk_node, softmax_node, out_node] {
            let node_name = node.name.clone();
            self.node_order.push(node_name.clone());
            self.nodes.insert(node_name, node);
        }

        debug!(
            "Decomposed SelfAttention '{}' into {}/qk + {}/softmax + {} (scale={})",
            name, name, name, name, scale,
        );

        Ok(())
    }

    /// Set the output node.
    pub fn set_output(&mut self, name: impl Into<String>) {
        self.invalidate_forward_linear_cache();
        self.output_node = name.into();
        self.cut_fold_scope = crate::beta_crown::bab_cuts::CutFoldScope::fresh();
        self.invalidate_dispatch_plan_cache();
    }

    /// The output node name.
    pub fn output_name(&self) -> &str {
        &self.output_node
    }

    /// Legacy compatibility alias for [`Self::output_name`].
    #[deprecated(note = "use output_name")]
    pub fn get_output_name(&self) -> &str {
        self.output_name()
    }

    /// A node by name.
    pub fn node(&self, name: &str) -> Option<&GraphNode> {
        self.nodes.get(name)
    }

    /// Legacy compatibility alias for [`Self::node`].
    #[deprecated(note = "use node")]
    pub fn get_node(&self, name: &str) -> Option<&GraphNode> {
        self.node(name)
    }

    /// Check if a node with the given name exists.
    #[inline]
    pub fn contains_node(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }

    /// Get all node names in insertion order.
    ///
    /// **Note:** For execution ordering, use [`Self::exec_order()`] when a
    /// borrowed cached slice is sufficient, or [`Self::topological_sort()`] if
    /// the caller needs an owned `Vec`.
    pub fn node_names(&self) -> &[String] {
        &self.node_order
    }

    /// Number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for GraphNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Layer, ReLULayer};

    #[test]
    #[should_panic(expected = "Duplicate node 'relu' (#2136)")]
    fn test_add_node_duplicate_name_panics_2959() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    }

    #[test]
    #[allow(deprecated)]
    fn test_graphnetwork_output_name_alias_matches_primary_4163() {
        let mut graph = GraphNetwork::new();
        graph.set_output("final");

        assert_eq!(graph.output_name(), "final");
        assert_eq!(graph.get_output_name(), graph.output_name());
    }

    #[test]
    fn test_elementwise_graph_is_batch_stack_safe() {
        // Pure element-wise / last-axis graph (ReLU-only stand-in for Gemm+ReLU
        // nets like ACAS-Xu): stacking domains on a leading axis is transparent,
        // so the batched IBP prescreen must remain enabled.
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        assert!(
            graph.is_input_split_batch_stack_safe(),
            "element-wise graph must be batch-stack safe"
        );
    }

    #[test]
    fn test_gather_graph_is_not_batch_stack_safe() {
        // A Gather over an absolute axis collides with the prepended batch axis
        // (the lsnc_relu quadrotor regression): the prescreen must be skipped.
        use crate::layers::GatherLayer;
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "gather",
            Layer::Gather(GatherLayer::new(0, None, vec![2])),
            vec!["relu".to_string()],
        ));
        assert!(
            !graph.is_input_split_batch_stack_safe(),
            "graph with an absolute-axis Gather must NOT be batch-stack safe"
        );
    }

    #[test]
    fn test_concat_graph_is_not_batch_stack_safe() {
        // Concat along an absolute axis is also unsafe to batch-stack.
        use crate::layers::ConcatLayer;
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("a", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::from_input("b", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "concat",
            Layer::Concat(ConcatLayer::new(0)),
            vec!["a".to_string(), "b".to_string()],
        ));
        assert!(
            !graph.is_input_split_batch_stack_safe(),
            "graph with an absolute-axis Concat must NOT be batch-stack safe"
        );
    }

    #[test]
    fn test_batchnorm_graph_is_not_batch_stack_safe() {
        // BatchNorm resolves its channel axis by value, so a prepended batch
        // axis whose extent equals the channel count would be scaled per-DOMAIN
        // instead of per-channel (the cgan ConvTranspose->BatchNorm shape):
        // the prescreen must be skipped.
        use crate::layers::BatchNormLayer;
        use ndarray::arr1;
        let bn = BatchNormLayer::from_scale_bias(
            arr1(&[1.0_f32, 10.0]).into_dyn(),
            arr1(&[0.0_f32, 0.0]).into_dyn(),
        )
        .expect("valid batchnorm");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("bn", Layer::BatchNorm(bn)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["bn".to_string()],
        ));
        assert!(
            !graph.is_input_split_batch_stack_safe(),
            "graph with a value-resolved-axis BatchNorm must NOT be batch-stack safe"
        );
    }
}
