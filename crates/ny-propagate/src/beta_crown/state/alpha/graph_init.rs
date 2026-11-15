// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

use ny_tensor::BoundedTensor;
use rustc_hash::FxHashMap;

use super::neuron::AlphaNeuronState;

/// Layer-name → neuron-index → α state, the pair of maps rebuilt for every BaB
/// child domain.
///
/// Deliberately NOT `std::collections::HashMap`. These keys are node names and
/// neuron indices that ny generates itself, so SipHash's DoS resistance buys
/// nothing, and rebuilding these maps per child made hashing the largest
/// identifiable cost in the serial path (measured 2026-08-08 on oval21
/// `base_kw`: `hash_one` 193, `insert` 152, `reserve_rehash` 59 samples).
///
/// Swapping the hasher changes iteration order. That is sound here because
/// every consumer is order-independent (`values_mut`, `values().all(..)`,
/// per-key mutation) — and because `std`'s `RandomState` is seeded per process,
/// so anything depending on this order would already be non-reproducible.
pub(crate) type AlphaNeuronMaps = FxHashMap<String, FxHashMap<usize, AlphaNeuronState>>;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::domain::NodeBoundsMap;
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

/// Graph-domain-specific α state for joint α-β optimization in graph BaB.
///
/// Like `DomainAlphaState` but keyed by `(node_name, neuron_idx)` instead of
/// `(layer_idx, neuron_idx)`, matching the graph network's node-based addressing.
///
/// This enables per-domain alpha optimization in the graph BaB path, which was
/// previously using only the fixed heuristic (alpha = 1 if u > -l, else 0).
///
/// # Reference
/// alpha-beta-CROWN: `auto_LiRPA/operators/relu.py` — optimizable slopes
#[derive(Debug, Clone)]
pub struct GraphDomainAlphaState {
    /// Lower-path per-neuron optimization state indexed by node_name → neuron_idx.
    /// Nested structure avoids String allocation per lookup: the outer map
    /// is keyed by `&str` (via `HashMap::get`), the inner by `usize`.
    /// Only stores entries for unstable, unconstrained ReLU neurons.
    pub(crate) neurons: AlphaNeuronMaps,
    /// STRUCTURAL staleness rule (#alpha-split-refresh, default-on): when a
    /// domain carries split constraints, its inherited per-neuron α values
    /// were optimized under ANCESTOR boxes that the splits have since
    /// re-clamped — measured on `mnist_fc 256x4`, the current-box heuristic
    /// beats those stale values. Set at MO-eval entry for split domains; the
    /// α array builder then derives slopes from the CURRENT box instead of
    /// the stale store. No knob: activation is structural (splits present).
    pub(crate) split_refresh: bool,
    /// Upper-path per-neuron optimization state, mirrored to `neurons`.
    pub(crate) upper_neurons: AlphaNeuronMaps,
}

impl GraphDomainAlphaState {
    /// Create empty alpha state.
    pub fn empty() -> Self {
        Self {
            split_refresh: false,
            neurons: AlphaNeuronMaps::default(),
            upper_neurons: AlphaNeuronMaps::default(),
        }
    }

    /// Whether stored values are delta-form offsets around the current-box
    /// heuristic (#alpha-split-refresh) rather than absolute slopes.
    pub fn split_refresh(&self) -> bool {
        self.split_refresh
    }

    /// #alpha-split-refresh at CHILD BIRTH (STRUCTURAL, default-on): flip this
    /// state to delta-form and neutralize every stored value to the zero
    /// offset (0.5 — composed slope = the CURRENT-box heuristic, see
    /// `build_alpha_array_from_node_map`).
    ///
    /// The receipted law is that stale inherited alpha measurably loses to the
    /// plain current-box heuristic: a parent's optimized values were ascended
    /// under ANCESTOR boxes that this domain's splits have since re-clamped
    /// (`mnist_fc 256x4` -7.046 receipts, REGRESSION doc). Until now the law
    /// ran only eval-locally (`objectives.rs` `alpha_local`) — the STORED
    /// child states kept absolute ancestor values, so the dense-spec adapter
    /// (depth > `beta_max_depth` children), the published-bound context of the
    /// analytical optimizer, and every grandchild consumed them verbatim.
    /// On `cora_2024 cifar10-set img398` that ascended-root-alpha inheritance
    /// (channel A of the steering-flow map) alone turns a 5.5-6.5 s unsat
    /// into a ~24 s timeout; suppressing it recovers unsat 3/3
    /// (REGRESSION doc, steering-flow discrimination receipts, 2026-08-30).
    ///
    /// Sound for any composed alpha in [0, 1] (`sanitize_alpha` clamp at every
    /// write, compose-time clamp in the array builder). Optimizer scratch is
    /// reset alongside: deltas step from a clean baseline.
    pub fn neutralize_for_split_refresh(&mut self) {
        self.split_refresh = true;
        for map in [&mut self.neurons, &mut self.upper_neurons] {
            for node in map.values_mut() {
                for neuron in node.values_mut() {
                    neuron.set_alpha(0.5);
                    neuron.reset_optimizer();
                }
            }
        }
    }

    /// Convert a global `GraphAlphaState` into per-neuron format for the
    /// batched backward kernel in input-split paths.
    ///
    /// In input-split verification, all sub-domains share the global α-CROWN
    /// state from the warmup pass. This converter creates a `GraphDomainAlphaState`
    /// with entries for neurons that were unstable at the root level. Sub-domain
    /// unstable neurons are always a subset of root unstable neurons (bounds
    /// get tighter after splitting), so this covers all relevant neurons.
    ///
    /// The batched backward core's `build_alpha_array` handles stable/unstable
    /// classification at runtime using per-domain pre-activation bounds, so
    /// stable neurons automatically get their correct slopes (0 or 1) regardless
    /// of the stored alpha values.
    ///
    /// # Reference
    /// alpha-beta-CROWN: `complete_verifier/input_split/branching_domains.py` —
    ///   "fix_interm_bounds" pattern shares root alpha across all sub-domains.
    pub fn from_global_alpha_state_for_input_split(
        global: &crate::bounds::GraphAlphaState,
    ) -> Self {
        // Pre-sized from the node counts we already know, so neither outer map
        // rehashes while filling. `reserve_rehash` was 59 samples in the
        // 2026-08-08 oval21 profile purely from growing these from empty.
        let mut neurons =
            AlphaNeuronMaps::with_capacity_and_hasher(global.alphas.len(), Default::default());
        let mut upper_neurons = AlphaNeuronMaps::with_capacity_and_hasher(
            global.alphas_upper.len(),
            Default::default(),
        );

        for (node_name, alphas) in &global.alphas {
            let mask = global.unstable_mask.get(node_name);
            // Upper bound on the unstable count; the map is dropped if empty.
            let mut node_map =
                FxHashMap::with_capacity_and_hasher(alphas.len(), Default::default());
            for (i, &alpha_val) in alphas.iter().enumerate() {
                let is_unstable = mask
                    .map(|m| m.get(i).copied().unwrap_or(false))
                    .unwrap_or(true); // conservative: include if no mask
                if is_unstable {
                    node_map.insert(i, AlphaNeuronState::new(alpha_val));
                }
            }
            if !node_map.is_empty() {
                neurons.insert(node_name.clone(), node_map);
            }
        }

        for (node_name, alphas_upper) in &global.alphas_upper {
            let mask = global.unstable_mask.get(node_name);
            let mut node_map =
                FxHashMap::with_capacity_and_hasher(alphas_upper.len(), Default::default());
            for (i, &alpha_val) in alphas_upper.iter().enumerate() {
                let is_unstable = mask
                    .map(|m| m.get(i).copied().unwrap_or(false))
                    .unwrap_or(true);
                if is_unstable {
                    node_map.insert(i, AlphaNeuronState::new(alpha_val));
                }
            }
            if !node_map.is_empty() {
                upper_neurons.insert(node_name.clone(), node_map);
            }
        }

        Self {
            split_refresh: false,
            neurons,
            upper_neurons,
        }
    }

    /// Initialize α state from graph node bounds and split history.
    ///
    /// For each ReLU node, identifies unstable neurons (l < 0 < u) that are not
    /// constrained by the history, and initializes α with the standard heuristic.
    ///
    /// # Arguments
    /// * `graph` - The graph network containing ReLU nodes
    /// * `node_bounds` - Pre-activation bounds for each node
    /// * `history` - Split history with ReLU constraints
    /// * `input_bounds` - Input bounds for the network
    pub fn from_graph_bounds(
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, Arc<BoundedTensor>>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Self {
        Self::from_graph_borrowed_bounds(graph, node_bounds, history, input_bounds)
    }

    /// Borrowed-bounds face of [`Self::from_graph_bounds`].
    ///
    /// Root-only bounded experiments already own a certified
    /// `HashMap<String, BoundedTensor>`.  Building a second full Arc-backed map
    /// merely to initialize advisory alpha state can consume a material part of
    /// a sub-second private budget.  This helper preserves the exact
    /// enumeration and heuristic initialization while accepting either owned
    /// or Arc-backed bounds without cloning tensor storage.
    pub(crate) fn from_graph_borrowed_bounds<V: Borrow<BoundedTensor>>(
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, V>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Self {
        Self::from_graph_bounds_lookup(graph, history, input_bounds, |name| {
            node_bounds.get(name).map(Borrow::borrow)
        })
    }

    fn from_graph_bounds_lookup<'a>(
        graph: &GraphNetwork,
        history: &GraphSplitHistory,
        input_bounds: &'a BoundedTensor,
        mut get_bound: impl FnMut(&str) -> Option<&'a BoundedTensor>,
    ) -> Self {
        let mut state = Self::empty();
        // One node's qualifying neurons, staged so the two outer maps are
        // touched ONCE per node instead of twice per neuron (see the insertion
        // block below). Reused across nodes, so it keeps the capacity the widest
        // node has needed so far and never allocates per node — but it is not a
        // single allocation either: `Vec` grows by doubling, so it reallocates
        // O(log w) times climbing to the widest node's qualifying-neuron count w.
        let mut staged: Vec<(usize, AlphaNeuronState)> = Vec::new();
        // The node's ReLU-constrained neuron indices, ascending, refilled once
        // per node at the site below. Reused across nodes on the same terms as
        // `staged`; it holds at most one entry per ReLU constraint in the
        // history — BaB depth — against the node's unstable-neuron count.
        let mut constrained_idx: Vec<usize> = Vec::new();

        for node_name in graph.node_names() {
            let node = match graph.node(node_name) {
                Some(n) => n,
                None => continue,
            };

            // Only process ReLU nodes
            if !matches!(node.layer, Layer::ReLU(_)) {
                continue;
            }

            // Skip nodes with empty inputs instead of fabricating NETWORK_INPUT.
            let pre_name = match node.inputs.first() {
                Some(s) => s.as_str(),
                None => {
                    tracing::warn!(node = %node_name, "ReLU node has empty inputs — skipping");
                    continue;
                }
            };
            let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
                input_bounds
            } else {
                match get_bound(pre_name) {
                    Some(bounds) => bounds,
                    None => continue,
                }
            };

            // Resolve the node's ReLU constraints ONCE per node. The loop below
            // called `is_constrained` (`branching/graph_history.rs:172`) once
            // per UNSTABLE pre-activation element — the stable-neuron skip runs
            // first, so not once per element — and every such call hashed
            // `node_name` against the two-level
            // `relu_lookup: HashMap<String, HashMap<usize, bool>>` (:33). That
            // is `std`'s `HashMap` (`:9`), i.e. a full SipHash of the name's
            // bytes, before the inner `usize` probe. What is left per unstable
            // neuron is a binary search over `constrained_idx`.
            //
            // `constrained_idx` is a SUPERSET of this node's `relu_lookup` keys,
            // never a subset, and `is_constrained` still makes the final call —
            // so the skip decision, and hence this state's contents, is exactly
            // what it was. The superset direction holds because `relu_lookup` is
            // a PRIVATE field with exactly two writers:
            // `GraphSplitHistory::new()`/`derive(Default)` (empty) and
            // `add_constraint` (:117-118), which pushes the same constraint onto
            // the public `constraints` vec (:24) in the same call (:122).
            // Nothing else can desynchronize them: there is no
            // `GraphSplitHistory` struct literal outside its own module (the
            // private field forbids one), the type derives no serde impl, and
            // the only non-test code that touches `constraints` is that single
            // `push` — the two `constraints[0].node_name = …` sites,
            // `engine/graph/resident_bab/operands.rs:1157` and `:1197`, are both
            // inside the `#[cfg(test)] mod tests` that opens at :746.
            //
            // GenBaB constraints are correctly absent: they land in
            // `genbab_constraints`/`genbab_lookup` (:26, :38), which
            // `is_constrained` never reads, so a GenBaB-only node yields an
            // empty `constrained_idx` and the old exact lookup would have missed
            // it too. The two stay in agreement rather than one covering for the
            // other.
            constrained_idx.clear();
            constrained_idx.extend(
                history
                    .constraints
                    .iter()
                    .filter(|c| c.node_name() == node_name.as_str())
                    .map(|c| c.neuron_idx()),
            );
            constrained_idx.sort_unstable();

            // #flatten-to-read-a-scalar: this used to call `pre_bounds.flatten()`,
            // which COPIES both endpoints into fresh `Vec<f32>`s
            // (`ny-tensor bounded_tensor/core/shape_ops.rs:130-138`) purely to
            // read two scalars per neuron. `flatten` fills those vectors with
            // `self.lower.iter()` — `TrackedArrayD` derefs to `ArrayD<f32>`
            // (`core/allocation_provenance.rs:217-224`), and `lower()` hands back
            // that same array (`core/mod.rs:110-112`) — then wraps them with
            // `from_parts_with_l2` (`core/mod.rs:162-172`), which is a bare
            // struct literal: no repair, no clamp, no arithmetic. Both sides
            // therefore go through the same `ArrayD::iter()`, so this zip yields
            // the same f32 values in the same order the flattened copy held, for
            // any layout. The sibling `find_unstable_graph_neurons_multi_bounded`
            // (`engine/branching/graph.rs`) reads its endpoints the same way;
            // `beta_crown/domain/multi_objective.rs:422-452` retired the same
            // `flatten()` call by a different route (an indexed
            // `as_slice()`/`nth()` read).
            //
            // Zip length caveat, stated precisely because it is a behaviour
            // change on one path: this scans `min(lower.len(), upper.len())`,
            // whereas the old loop ran to `pre_flat.len()` = `lower.len()` and
            // indexed `upper`. Those agree only while the two endpoints have
            // equal shapes, and equal shapes are a CALLER CONTRACT here, not a
            // construction guarantee — `BoundedTensor::new`
            // (`ny-tensor .../core/constructors.rs:142-148`) does reject a
            // mismatch in every build, but the crate-internal
            // `from_parts_unchecked` (`core/mod.rs:136-143`) enforces it with
            // `debug_assert_eq!` only and `from_parts_with_l2` not at all. So in
            // a release build with a violated contract and a shorter `upper`,
            // the old code panicked on the index and this zip stops short
            // instead, leaving the tail neurons out of the α state.
            staged.clear();
            for (neuron_idx, (&l, &u)) in pre_bounds
                .lower()
                .iter()
                .zip(pre_bounds.upper().iter())
                .enumerate()
            {
                // Skip stable neurons
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Skip constrained neurons (they use fixed slopes 0 or 1)
                if !constrained_idx.is_empty() && constrained_idx.binary_search(&neuron_idx).is_ok()
                {
                    record_exact_constraint_lookup();
                    if history.is_constrained(node_name, neuron_idx).is_some() {
                        continue;
                    }
                }

                // Unstable and unconstrained: initialize α with heuristic
                let alpha = if u > -l { 1.0 } else { 0.0 };
                staged.push((neuron_idx, AlphaNeuronState::new(alpha)));
            }

            // Two `entry(node_name.clone())` calls per node, not per neuron. The
            // early `continue` is load-bearing, not a micro-optimization: a node
            // with no qualifying neuron must stay ABSENT from both maps rather
            // than carry an empty inner map, because the node-KEY SET is read
            // directly — `active_set_gpu_alpha.rs` fingerprints
            // `lower_node_names.len()` and each name's bytes, and
            // `critical_gpu_alpha.rs` collects `neurons().keys()` — so an eager
            // hoist would silently change that fingerprint. Each map still
            // receives its inserts in ascending `neuron_idx`, so its bucket
            // layout, and hence its iteration order, is unchanged; only the
            // interleaving BETWEEN the two maps differs, which neither map can
            // observe.
            if staged.is_empty() {
                continue;
            }
            let lower_map = state.neurons.entry(node_name.clone()).or_default();
            for &(neuron_idx, neuron_state) in &staged {
                lower_map.insert(neuron_idx, neuron_state);
            }
            let upper_map = state.upper_neurons.entry(node_name.clone()).or_default();
            for &(neuron_idx, neuron_state) in &staged {
                upper_map.insert(neuron_idx, neuron_state);
            }
        }

        state
    }

    /// Initialize from root-level optimized α-CROWN state.
    ///
    /// Transfers the gradient-optimized alpha values from `GraphAlphaState`
    /// (used during root-level α-CROWN in `collect_alpha_crown_bounds_dag`)
    /// into a sparse `GraphDomainAlphaState` for the BaB root domain.
    ///
    /// Without this, root domain alpha would be re-initialized from the
    /// `u > -l` heuristic, discarding the SPSA/Adam-optimized values.
    ///
    /// # Arguments
    /// * `root_alpha` - Optimized `GraphAlphaState` from `collect_alpha_crown_bounds_dag`
    /// * `graph` - Graph network for identifying ReLU nodes
    /// * `node_bounds` - Pre-activation bounds per node
    /// * `history` - Split history (empty for root domain)
    /// * `input_bounds` - Input bounds for the network
    pub fn from_root_alpha_state(
        root_alpha: &crate::bounds::GraphAlphaState,
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, Arc<BoundedTensor>>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Self {
        Self::from_root_alpha_state_borrowed(root_alpha, graph, node_bounds, history, input_bounds)
    }

    /// Borrowed-bounds face of [`Self::from_root_alpha_state`].
    ///
    /// The alpha bridge used by bounded root experiments is already dense and
    /// full-spatial.  Converting it back through this path provides a cheap
    /// round-trip identity check without cloning the certified bound map.
    pub(crate) fn from_root_alpha_state_borrowed<V: Borrow<BoundedTensor>>(
        root_alpha: &crate::bounds::GraphAlphaState,
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, V>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Self {
        let expand_channel_only =
            std::env::var("NY_ALPHA_INHERIT_EXPAND").ok().as_deref() == Some("1");
        Self::from_root_alpha_state_borrowed_with_expansion(
            root_alpha,
            graph,
            node_bounds,
            history,
            input_bounds,
            expand_channel_only,
        )
    }

    /// Convert root α while explicitly expanding channel-only convolutional
    /// parameters to the flattened per-neuron layout.
    ///
    /// This is deliberately separate from `NY_ALPHA_INHERIT_EXPAND`: bounded
    /// paired-state experiments need a correctly indexed candidate without
    /// changing the established heuristic/default transport globally.
    pub(crate) fn try_from_root_alpha_state_borrowed_expanded<V: Borrow<BoundedTensor>>(
        root_alpha: &crate::bounds::GraphAlphaState,
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, V>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Option<Self> {
        // Lower and upper paths are one candidate identity. Refuse a partial
        // map rather than silently pairing one optimized path with a heuristic
        // or with the other path's parameters.
        if !root_alpha.alphas.keys().eq(root_alpha.alphas_upper.keys())
            || root_alpha
                .spatial_shapes
                .keys()
                .any(|node_name| !root_alpha.alphas.contains_key(node_name))
        {
            return None;
        }

        // Validate and expand every lower/upper ReLU array before constructing
        // or mutating the sparse candidate. Exact tensor shape is part of the
        // identity: equal flattened products are not interchangeable because
        // channel-only expansion depends on the leading channel dimension.
        let mut expanded_lower = HashMap::new();
        let mut expanded_upper = HashMap::new();
        for (node_name, lower) in &root_alpha.alphas {
            let node = graph.nodes.get(node_name)?;
            if !matches!(node.layer, Layer::ReLU(_)) {
                return None;
            }
            let pre_name = node.inputs.first()?;
            let pre_bounds = if pre_name == NETWORK_INPUT {
                input_bounds
            } else {
                node_bounds.get(pre_name)?.borrow()
            };
            let upper = root_alpha.alphas_upper.get(node_name)?;
            let lower = root_alpha.try_expand_alpha(node_name, lower, pre_bounds.shape())?;
            let upper = root_alpha.try_expand_alpha(node_name, upper, pre_bounds.shape())?;
            if lower.len() != pre_bounds.len() || upper.len() != pre_bounds.len() {
                return None;
            }
            expanded_lower.insert(node_name.as_str(), lower);
            expanded_upper.insert(node_name.as_str(), upper);
        }

        // Only now create the heuristic state and overwrite it. Any malformed
        // array above returns without a partially updated candidate escaping.
        let mut state = Self::from_graph_borrowed_bounds(graph, node_bounds, history, input_bounds);
        for (node_name, neuron_map) in &mut state.neurons {
            let alpha_arr = expanded_lower.get(node_name.as_str());
            if let Some(alpha_arr) = alpha_arr {
                for (&neuron_idx, neuron_state) in neuron_map.iter_mut() {
                    let &alpha = alpha_arr.get(neuron_idx)?;
                    neuron_state.set_alpha(alpha);
                }
            }
        }

        for (node_name, neuron_map) in &mut state.upper_neurons {
            let alpha_arr = expanded_upper.get(node_name.as_str());
            if let Some(alpha_arr) = alpha_arr {
                for (&neuron_idx, neuron_state) in neuron_map.iter_mut() {
                    let &alpha = alpha_arr.get(neuron_idx)?;
                    neuron_state.set_alpha(alpha);
                }
            }
        }

        Some(state)
    }

    fn from_root_alpha_state_borrowed_with_expansion<V: Borrow<BoundedTensor>>(
        root_alpha: &crate::bounds::GraphAlphaState,
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, V>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
        expand_channel_only: bool,
    ) -> Self {
        // Start with heuristic initialization (handles neuron enumeration,
        // constraint filtering, and stable neuron skipping).
        let mut state = Self::from_graph_borrowed_bounds(graph, node_bounds, history, input_bounds);

        // #hard-six α-inherit-expand (dark, NY_ALPHA_INHERIT_EXPAND=1):
        // when the warmup ran CHANNEL-ONLY α (`full_conv_alpha: false`, e.g.
        // the cifar100_2024 preset — 1188 shared params for ~24k unstable
        // neurons), `root_alpha.alphas[node]` has length C while `neuron_idx`
        // here is the FLAT per-neuron index into [C, H, W]. The historical
        // code below then (a) silently keeps the heuristic α for every neuron
        // with flat index ≥ C (the warmup optimum never reaches BaB) and
        // (b) mis-indexes the first C neurons with OTHER channels' α values.
        // With the gate on, channel-only arrays are spatially broadcast to
        // per-neuron before the override, so every unstable neuron seeds from
        // its own channel's warmup-optimized α. Unset ⇒ byte-identical.
        // SOUND either way: any α ∈ [0,1] yields a valid CROWN relaxation;
        // this only changes the ascent's starting point.
        // Override heuristic alpha with optimized values from root α-CROWN.
        // GraphAlphaState stores dense Array1 per node name; GraphDomainAlphaState
        // stores sparse node_name → neuron_idx entries.
        for (node_name, neuron_map) in &mut state.neurons {
            if let Some(alpha_arr) = root_alpha.alphas.get(node_name) {
                let expanded = (expand_channel_only
                    && root_alpha.spatial_shape(node_name).is_some())
                .then(|| root_alpha.expand_alpha(node_name, alpha_arr));
                let alpha_arr = expanded.as_ref().unwrap_or(alpha_arr);
                for (&neuron_idx, neuron_state) in neuron_map.iter_mut() {
                    if neuron_idx < alpha_arr.len() {
                        neuron_state.set_alpha(alpha_arr[neuron_idx]);
                    }
                }
            }
        }

        for (node_name, neuron_map) in &mut state.upper_neurons {
            let alpha_arr = root_alpha
                .alphas_upper
                .get(node_name)
                .or_else(|| root_alpha.alphas.get(node_name));
            if let Some(alpha_arr) = alpha_arr {
                let expanded = (expand_channel_only
                    && root_alpha.spatial_shape(node_name).is_some())
                .then(|| root_alpha.expand_alpha(node_name, alpha_arr));
                let alpha_arr = expanded.as_ref().unwrap_or(alpha_arr);
                for (&neuron_idx, neuron_state) in neuron_map.iter_mut() {
                    if neuron_idx < alpha_arr.len() {
                        neuron_state.set_alpha(alpha_arr[neuron_idx]);
                    }
                }
            }
        }

        state
    }

    /// Initialize from parent state, inheriting optimized alpha values.
    ///
    /// For neurons that were also unstable in the parent domain, copies the
    /// parent's optimized alpha (warm start). For newly unstable neurons,
    /// uses the heuristic initialization.
    pub fn from_parent(
        parent: &GraphDomainAlphaState,
        graph: &GraphNetwork,
        node_bounds: &HashMap<String, Arc<BoundedTensor>>,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Self {
        let state = Self::from_graph_bounds(graph, node_bounds, history, input_bounds);
        Self::inherit_parent_values(parent, state)
    }

    pub(crate) fn from_parent_node_bounds_map(
        parent: &GraphDomainAlphaState,
        graph: &GraphNetwork,
        node_bounds: &NodeBoundsMap,
        history: &GraphSplitHistory,
        input_bounds: &BoundedTensor,
    ) -> Self {
        let state = Self::from_graph_bounds_lookup(graph, history, input_bounds, |name| {
            node_bounds.get(name).map(AsRef::as_ref)
        });
        Self::inherit_parent_values(parent, state)
    }

    fn inherit_parent_values(parent: &GraphDomainAlphaState, mut state: Self) -> Self {
        // Warm-start: inherit parent's optimized alpha for matching neurons
        for (node_name, parent_neuron_map) in &parent.neurons {
            if let Some(child_neuron_map) = state.neurons.get_mut(node_name) {
                for (&neuron_idx, parent_neuron) in parent_neuron_map {
                    if let Some(child_neuron) = child_neuron_map.get_mut(&neuron_idx) {
                        child_neuron.set_alpha(parent_neuron.alpha);
                        // Reset optimizer state for fresh optimization in child domain
                    }
                }
            }
        }

        for (node_name, parent_neuron_map) in &parent.neurons {
            if let Some(child_neuron_map) = state.upper_neurons.get_mut(node_name) {
                for (&neuron_idx, parent_neuron) in parent_neuron_map {
                    if let Some(child_neuron) = child_neuron_map.get_mut(&neuron_idx) {
                        let upper_alpha = parent
                            .upper_neurons
                            .get(node_name)
                            .and_then(|m| m.get(&neuron_idx))
                            .map_or(parent_neuron.alpha(), AlphaNeuronState::alpha);
                        child_neuron.set_alpha(upper_alpha);
                    }
                }
            }
        }

        state
    }
}

// Test-only count of the EXACT `GraphSplitHistory::is_constrained` lookups
// performed by [`GraphDomainAlphaState::from_graph_bounds_lookup`].
//
// The per-node constraint resolution in that builder is invisible from the
// state it returns — both maps hold exactly the entries they always held — so
// reverting it to the historical per-neuron `is_constrained` call would break
// nothing that a contents-comparing test could see. This counter is what fails
// instead: the fast lane reaches the exact map only for indices the node's own
// resolved constraint list flagged, while the historical loop reached it once
// per unstable neuron.
//
// Thread-local, so tests running in parallel cannot observe each other's
// counts, and `#[cfg(test)]`, so the shipped builder carries no counter at all.
#[cfg(test)]
thread_local! {
    static TEST_EXACT_CONSTRAINT_LOOKUPS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_exact_constraint_lookup() {
    TEST_EXACT_CONSTRAINT_LOOKUPS.with(|count| {
        count.set(count.get().saturating_add(1));
    });
}

// Production no-op twin of the counter above, so the call site is an ordinary
// statement in every build. Without it the `#[cfg(test)]` attribute would leave
// the release build with an `if` whose sole statement is another `if`, which is
// `clippy::collapsible_if`; an empty `#[inline]` function optimizes away
// instead.
#[cfg(not(test))]
#[inline(always)]
fn record_exact_constraint_lookup() {}

#[cfg(test)]
fn reset_test_exact_constraint_lookups() {
    TEST_EXACT_CONSTRAINT_LOOKUPS.with(|count| count.set(0));
}

#[cfg(test)]
fn test_exact_constraint_lookups() -> usize {
    TEST_EXACT_CONSTRAINT_LOOKUPS.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use ny_tensor::BoundedTensor;

    use super::{
        reset_test_exact_constraint_lookups, test_exact_constraint_lookups, GraphDomainAlphaState,
    };
    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
    use crate::layers::ReLULayer;
    use crate::{GraphNetwork, GraphNode, Layer};

    /// A one-ReLU graph whose producer is an ordinary named node, so the builder
    /// takes the `node_bounds` branch rather than the network-input one.
    fn single_relu_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph
    }

    fn all_unstable(width: usize) -> BoundedTensor {
        BoundedTensor::new(
            ndarray::Array1::from_elem(width, -1.0_f32).into_dyn(),
            ndarray::Array1::from_elem(width, 1.0_f32).into_dyn(),
        )
        .expect("all-unstable pre-activation bounds")
    }

    fn unit_input() -> BoundedTensor {
        BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("input bounds")
    }

    /// FAIL-CLOSED ENGAGEMENT PROOF for the per-node constraint resolution.
    ///
    /// Drives the shipped `from_graph_bounds` — not a private copy — over 64
    /// unstable neurons on a node carrying ONE ReLU constraint, and asserts the
    /// exact `is_constrained` map was consulted exactly once. The historical
    /// loop consulted it 64 times, so restoring the per-neuron call fails this
    /// assertion while leaving the alpha state, asserted below it, untouched.
    #[test]
    fn from_graph_bounds_resolves_constraints_once_per_node() {
        const WIDTH: usize = 64;
        const CONSTRAINED: usize = 7;

        let graph = single_relu_graph();
        let mut node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
        node_bounds.insert("pre".to_string(), Arc::new(all_unstable(WIDTH)));

        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("relu0".to_string(), CONSTRAINED, true, 1.0)
                .expect("finite score"),
        );

        reset_test_exact_constraint_lookups();
        let state =
            GraphDomainAlphaState::from_graph_bounds(&graph, &node_bounds, &history, &unit_input());
        let lookups = test_exact_constraint_lookups();

        let lower = state.neurons.get("relu0").expect("relu0 alpha entries");
        assert_eq!(
            lower.len(),
            WIDTH - 1,
            "the constrained neuron is the only one excluded"
        );
        assert!(
            !lower.contains_key(&CONSTRAINED),
            "relu0[{CONSTRAINED}] is constrained and must carry no alpha state"
        );
        assert_eq!(
            state
                .upper_neurons
                .get("relu0")
                .expect("relu0 upper alpha entries")
                .len(),
            WIDTH - 1,
            "the upper map mirrors the lower map"
        );

        assert_eq!(
            lookups, 1,
            "the builder must resolve relu0's constraints once per node and reach \
             is_constrained only for the resolved index; {lookups} exact lookups \
             over {WIDTH} unstable neurons means the per-neuron lookup is back"
        );
    }

    /// The zero-constraint case must never touch the exact map at all — the
    /// common shape in BaB, where most ReLU nodes carry no split of their own.
    #[test]
    fn from_graph_bounds_skips_the_exact_map_for_an_unconstrained_node() {
        const WIDTH: usize = 32;

        let graph = single_relu_graph();
        let mut node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
        node_bounds.insert("pre".to_string(), Arc::new(all_unstable(WIDTH)));

        // A constraint on a DIFFERENT node: `constraints` is non-empty, so the
        // resolution cannot come up empty merely because the history is.
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("relu_other".to_string(), 3, false, 1.0)
                .expect("finite score"),
        );

        reset_test_exact_constraint_lookups();
        let state =
            GraphDomainAlphaState::from_graph_bounds(&graph, &node_bounds, &history, &unit_input());
        let lookups = test_exact_constraint_lookups();

        assert_eq!(
            state
                .neurons
                .get("relu0")
                .expect("relu0 alpha entries")
                .len(),
            WIDTH,
            "no neuron of relu0 is constrained"
        );
        assert_eq!(
            lookups, 0,
            "a node with no ReLU constraint of its own must never reach the \
             exact map; got {lookups} lookups"
        );
    }

    /// The endpoint read must follow `ArrayD::iter()` — logical row-major, which
    /// is exactly what the retired `flatten()` collected — and NOT the raw
    /// buffer. `reversed_axes()` keeps the original buffer with transposed
    /// strides, so `as_slice()` is `None` and buffer order differs from logical
    /// order; a substitution that read memory order would key alpha state by
    /// different indices, and would pick different heuristic slopes.
    #[test]
    fn from_graph_bounds_reads_logical_order_on_a_transposed_tensor() {
        // Row-major 2x3, then transposed to 3x2 without copying.
        //   lower buffer [-2, 1, -1,  1, -1, 1] -> iter [-2,  1,  1, -1, -1, 1]
        //   upper buffer [ 3, 2,  3,  2,  1, 2] -> iter [ 3,  2,  2,  1,  3, 2]
        // so l < 0 < u holds at logical indices 0, 3, 4, and the `u > -l`
        // heuristic gives alpha 1.0 at index 0 (3 > 2), 0.0 at 3 (1 <= 1) and
        // 1.0 at 4 (3 > 1). Reading the BUFFER instead would make 0, 2, 4
        // unstable — a different index set, so the assertion below separates
        // the two.
        let lower =
            ndarray::Array2::from_shape_vec((2, 3), vec![-2.0_f32, 1.0, -1.0, 1.0, -1.0, 1.0])
                .expect("2x3 lower")
                .reversed_axes()
                .into_dyn();
        let upper = ndarray::Array2::from_shape_vec((2, 3), vec![3.0_f32, 2.0, 3.0, 2.0, 1.0, 2.0])
            .expect("2x3 upper")
            .reversed_axes()
            .into_dyn();
        assert!(
            lower.as_slice().is_none(),
            "the test needs a non-standard layout for the read to be observable"
        );
        let pre = BoundedTensor::new(lower, upper).expect("transposed pre-activation bounds");

        // Reference: what the retired `flatten()` read, computed the old way.
        let flat = pre.flatten();
        let mut reference: Vec<(usize, f32)> = Vec::new();
        for idx in 0..flat.len() {
            let l = flat.lower()[[idx]];
            let u = flat.upper()[[idx]];
            if l < 0.0 && u > 0.0 {
                reference.push((idx, if u > -l { 1.0 } else { 0.0 }));
            }
        }

        let graph = single_relu_graph();
        let mut node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
        node_bounds.insert("pre".to_string(), Arc::new(pre));
        let history = GraphSplitHistory::new();

        let state =
            GraphDomainAlphaState::from_graph_bounds(&graph, &node_bounds, &history, &unit_input());
        let lower_map = state.neurons.get("relu0").expect("relu0 alpha entries");

        let mut got: Vec<(usize, f32)> = lower_map
            .iter()
            .map(|(&idx, neuron)| (idx, neuron.alpha()))
            .collect();
        got.sort_by_key(|(idx, _)| *idx);

        assert_eq!(
            got,
            vec![(0, 1.0), (3, 0.0), (4, 1.0)],
            "logical (iter) order, not buffer order"
        );
        assert_eq!(got, reference, "must match what flatten() would have read");
    }
}
