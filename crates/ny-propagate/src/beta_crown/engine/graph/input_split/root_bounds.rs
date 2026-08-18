// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Root intermediate-bound warmup helpers for graph input split.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::beta_crown::config::BetaCrownConfig;
use crate::bounds::{GradientMethod, GraphAlphaState};
use crate::network::PrecomputedAlphaReferenceBounds;
use crate::GraphNetwork;

type RootBoundsValue = (
    Option<HashMap<String, BoundedTensor>>,
    Option<GraphAlphaState>,
);

/// Clone the numeric alpha state bit-for-bit while detaching its mutable,
/// cost-only GPU-suffix negative cache. Ordinary `GraphAlphaState::clone`
/// intentionally shares that `Arc`; a first restart may populate it during
/// BaB, which must not mutate the frozen root snapshot held for restart #2.
fn clone_alpha_state_for_restart_cache(state: &GraphAlphaState) -> GraphAlphaState {
    let mut cloned = state.clone();
    let ineligible = state
        .gpu_suffix_ineligible
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    cloned.gpu_suffix_ineligible = Arc::new(std::sync::RwLock::new(ineligible));
    cloned
}

fn clone_root_bounds_value(value: &RootBoundsValue) -> RootBoundsValue {
    (
        value.0.clone(),
        value.1.as_ref().map(clone_alpha_state_for_restart_cache),
    )
}

/// Exact identity of one restart-reusable root collection.
///
/// `coverage` is the collision-proof byte string used by the graph's existing
/// CROWN-IBP cache: every input lower/upper f32 bit, input shape, graph coverage
/// descriptor, patches policy, and engine-presence bit. `graph_scope` prevents
/// a same-shaped/dimensioned foreign graph from ever matching. `options` adds
/// every root-collection option not already covered there. `spec` scopes the
/// cache to the exact packed disjunctive property, even though the current root
/// intermediate collector is spec-independent.
#[derive(Clone, PartialEq, Eq)]
struct RootBoundsCacheKey {
    graph_scope: crate::beta_crown::bab_cuts::CutFoldScope,
    coverage: Arc<[u8]>,
    options: Arc<[u8]>,
    spec: Arc<[u8]>,
}

#[derive(Clone)]
struct RootBoundsCacheEntry {
    key: RootBoundsCacheKey,
    value: Arc<RootBoundsValue>,
}

#[derive(Clone)]
struct TypedReferenceMapCacheEntry {
    key: RootBoundsCacheKey,
    value: Arc<PrecomputedAlphaReferenceBounds>,
}

/// One-entry cache shared only by the deterministic restarts of ONE top-level
/// grouped-disjunctive verification call.
///
/// The CLI creates a fresh instance for explicitly typed cGAN roots or behind
/// `NY_DISJUNCTIVE_RESTART_ROOT_CACHE=1`; it is dropped when that top-level
/// call returns. A cache hit is exact reuse of a previously certified map and
/// `GraphAlphaState`, never a newly computed or widened approximation. Any
/// identity mismatch, non-deterministic SPSA/supplement root configuration,
/// serialization failure, or poisoned lock fails closed to the original
/// collection path.
pub(crate) struct InputSplitRootBoundsCache {
    spec_identity: Arc<[u8]>,
    overall_deadline: Option<Instant>,
    entry: Mutex<Option<RootBoundsCacheEntry>>,
    /// Deterministic typed reference map, kept separate from restart-sensitive
    /// alpha/optimizer state. Used only when the whole-result cache must bypass
    /// an RNG-consuming root.
    typed_reference_entry: Mutex<Option<TypedReferenceMapCacheEntry>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
    collections: AtomicUsize,
    typed_reference_hits: AtomicUsize,
    typed_reference_misses: AtomicUsize,
    typed_reference_collections: AtomicUsize,
}

impl InputSplitRootBoundsCache {
    pub(crate) fn new(spec_identity: Vec<u8>, overall_deadline: Option<Instant>) -> Self {
        Self {
            spec_identity: Arc::from(spec_identity),
            overall_deadline,
            entry: Mutex::new(None),
            typed_reference_entry: Mutex::new(None),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            collections: AtomicUsize::new(0),
            typed_reference_hits: AtomicUsize::new(0),
            typed_reference_misses: AtomicUsize::new(0),
            typed_reference_collections: AtomicUsize::new(0),
        }
    }

    pub(crate) fn deadline_matches(&self, deadline: Option<Instant>) -> bool {
        self.overall_deadline == deadline
    }

    pub(crate) fn spec_matches(&self, spec_identity: &[u8]) -> bool {
        self.spec_identity.as_ref() == spec_identity
    }

    pub(crate) fn hits(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    pub(crate) fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }

    fn note_collection(&self) -> usize {
        self.collections.fetch_add(1, Ordering::Relaxed) + 1
    }

    #[cfg(test)]
    fn collections(&self) -> usize {
        self.collections.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn typed_reference_hits(&self) -> usize {
        self.typed_reference_hits.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn typed_reference_collections(&self) -> usize {
        self.typed_reference_collections.load(Ordering::Relaxed)
    }

    fn lookup(&self, key: &RootBoundsCacheKey) -> Option<RootBoundsValue> {
        let Ok(guard) = self.entry.lock() else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let Some(entry) = guard.as_ref() else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        if entry.key != *key {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(clone_root_bounds_value(&entry.value))
    }

    fn store(&self, key: RootBoundsCacheKey, value: &RootBoundsValue) {
        let Ok(mut guard) = self.entry.lock() else {
            return;
        };
        *guard = Some(RootBoundsCacheEntry {
            key,
            value: Arc::new(clone_root_bounds_value(value)),
        });
    }

    fn lookup_typed_reference(
        &self,
        key: &RootBoundsCacheKey,
    ) -> Option<PrecomputedAlphaReferenceBounds> {
        let Ok(guard) = self.typed_reference_entry.lock() else {
            self.typed_reference_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let Some(entry) = guard.as_ref() else {
            self.typed_reference_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        if entry.key != *key {
            self.typed_reference_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.typed_reference_hits.fetch_add(1, Ordering::Relaxed);
        Some((*entry.value).clone())
    }

    fn note_typed_reference_collection(&self) -> usize {
        self.typed_reference_collections
            .fetch_add(1, Ordering::Relaxed)
            + 1
    }

    fn store_typed_reference(
        &self,
        key: RootBoundsCacheKey,
        value: &PrecomputedAlphaReferenceBounds,
    ) {
        let Ok(mut guard) = self.typed_reference_entry.lock() else {
            return;
        };
        *guard = Some(TypedReferenceMapCacheEntry {
            key,
            value: Arc::new(value.clone()),
        });
    }
}

fn push_len(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

/// Collision-proof packed disjunctive-spec identity: dimensions and every f32
/// threshold/objective bit are compared directly, never through a hash alone.
pub(crate) fn disjunctive_spec_identity(
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    clause_sizes: &[usize],
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"ny.disjunctive-root-cache.spec.v1\0");
    push_len(&mut bytes, objectives.len());
    for objective in objectives {
        push_len(&mut bytes, objective.len());
        for value in objective {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    push_len(&mut bytes, thresholds.len());
    for threshold in thresholds {
        bytes.extend_from_slice(&threshold.to_bits().to_le_bytes());
    }
    push_len(&mut bytes, clause_sizes.len());
    for size in clause_sizes {
        push_len(&mut bytes, *size);
    }
    bytes
}

/// Exact non-deadline identity of every option read by the graph alpha/root
/// collection surface. The producer's phase deadline is intentionally NOT a
/// semantic key: a certified map remains valid after time advances. The cache
/// itself is bound to the original shared absolute deadline and is handed to a
/// restart only when that exact deadline matches; a hit creates no fresh grace.
fn root_collection_options_identity(
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"ny.disjunctive-root-cache.options.v2\0");
    bytes.push(u8::from(config.use_alpha_crown));
    bytes.push(u8::from(config.use_forward_bounds));
    bytes.push(u8::from(config.use_crown_ibp));
    bytes.push(u8::from(config.verify_upper_bound));
    match config.crown_backward_layers {
        Some(value) => {
            bytes.push(1);
            push_len(&mut bytes, value);
        }
        None => bytes.push(0),
    }
    for value in [
        config.crown_ibp_per_node_floor_secs,
        config.crown_ibp_per_node_cap_secs,
    ] {
        match value {
            Some(value) => {
                bytes.push(1);
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            None => bytes.push(0),
        }
    }

    // JSON supplies a version-tolerant structural identity for every serialized
    // alpha option (including enum/string/vector shape). Append every current
    // floating field as raw IEEE bits as well, so even non-finite payloads and
    // `-0.0` can never alias through a textual representation. The two
    // #[serde(skip)] semantic fields are appended explicitly below.
    let alpha = serde_json::to_vec(&config.alpha_config).ok()?;
    push_len(&mut bytes, alpha.len());
    bytes.extend_from_slice(&alpha);
    for value in [
        config.alpha_config.learning_rate,
        config.alpha_config.lr_decay,
        config.alpha_config.tolerance,
        config.alpha_config.momentum,
        config.alpha_config.sparse_ratio,
        config.alpha_config.pilot_improvement_threshold,
        config.alpha_config.adam_beta1,
        config.alpha_config.adam_beta2,
        config.alpha_config.adam_epsilon,
        config.alpha_config.pruning_in_iteration_threshold,
        config.alpha_config.invprop.gamma_lr,
        config.alpha_config.start_save_best,
        config.alpha_config.reference_refresh_fraction,
    ] {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    match config.alpha_config.output_constraints.as_ref() {
        Some(constraints) => {
            bytes.push(1);
            let encoded = serde_json::to_vec(constraints).ok()?;
            push_len(&mut bytes, encoded.len());
            bytes.extend_from_slice(&encoded);
            push_len(&mut bytes, constraints.a_matrix.nrows());
            push_len(&mut bytes, constraints.a_matrix.ncols());
            for value in &constraints.a_matrix {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            push_len(&mut bytes, constraints.rhs.len());
            for value in &constraints.rhs {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        None => bytes.push(0),
    }
    match config.alpha_config.spec_early_exit.as_ref() {
        Some(spec) => {
            bytes.push(1);
            push_len(&mut bytes, spec.objective.len());
            for value in &spec.objective {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            bytes.extend_from_slice(&spec.threshold.to_bits().to_le_bytes());
            bytes.push(u8::from(spec.verify_upper_bound));
        }
        None => bytes.push(0),
    }
    // #root-alpha-margin: the ranking objective selects WHICH α the warmup returns,
    // so two requests differing only in their margin rows yield different root α
    // state. Without this discriminator the cache could serve one property's α for
    // another's rows — the single wrong-verdict vector in that change.
    //
    // The `None` arm still pushes a byte, so every identity shifts relative to the
    // pre-change build. That is harmless and deliberate: this key indexes an
    // in-process `HashMap` built fresh each run, so a changed key can only ever cause
    // a cold miss, never a stale hit. The version tag above is bumped in step so the
    // change is explicit rather than implied by a silent layout shift.
    match config.alpha_config.spec_ascent.as_ref() {
        Some(ascent) => {
            bytes.push(1);
            push_len(&mut bytes, ascent.rows.len());
            for row in &ascent.rows {
                push_len(&mut bytes, row.objective.len());
                for value in &row.objective {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                bytes.extend_from_slice(&row.threshold.to_bits().to_le_bytes());
                bytes.push(u8::from(row.verify_upper_bound));
            }
        }
        None => bytes.push(0),
    }

    // The exact engine object is part of the identity, not merely a boolean:
    // two engines can use different certified summation implementations.
    let engine_identity = engine
        .map(|engine| std::ptr::from_ref(engine).cast::<()>() as usize)
        .unwrap_or(0);
    bytes.extend_from_slice(&engine_identity.to_le_bytes());

    // Capture the complete NY runtime-option namespace, rather than a brittle
    // allow-list that could silently miss a newly added root-collection knob.
    // Also include numerical-backend/thread routing namespaces. Secrets and
    // unrelated process state are intentionally excluded. Values are compared
    // byte-for-byte after sorting; any non-Unicode name/value fails closed.
    let mut env_options = Vec::new();
    for (raw_name, raw_value) in std::env::vars_os() {
        let name = raw_name.into_string().ok()?;
        if name.starts_with("NY_")
            || name.starts_with("WGPU_")
            || name.starts_with("CUDA_")
            || name.starts_with("RAYON_")
            || name.starts_with("OMP_")
            || name.starts_with("MKL_")
            || name.starts_with("OPENBLAS_")
        {
            env_options.push((name, raw_value.into_string().ok()?));
        }
    }
    env_options.sort_unstable();
    push_len(&mut bytes, env_options.len());
    for (name, value) in env_options {
        push_len(&mut bytes, name.len());
        bytes.extend_from_slice(name.as_bytes());
        push_len(&mut bytes, value.len());
        bytes.extend_from_slice(value.as_bytes());
    }
    Some(bytes)
}

/// Root α collection is restart-invariant only when it cannot consume the
/// restart-offset RNG. Besides an explicit SPSA gradient method, the DAG alpha
/// optimizer uses SPSA supplements for MulBinary and optimizable
/// Sigmoid/Tanh/Sqrt/Reciprocal nodes even under analytic primary gradients.
/// Fail closed for those graphs whenever an optimization iteration can run.
fn root_collection_is_restart_deterministic(
    graph: &GraphNetwork,
    config: &BetaCrownConfig,
) -> bool {
    if !config.use_alpha_crown || config.alpha_config.iterations == 0 {
        return true;
    }
    if config.alpha_config.gradient_method == GradientMethod::Spsa {
        return false;
    }
    !graph.node_names().iter().any(|name| {
        graph.node(name).is_some_and(|node| {
            matches!(
                node.layer(),
                crate::Layer::MulBinary(_)
                    | crate::Layer::Sigmoid(_)
                    | crate::Layer::Tanh(_)
                    | crate::Layer::Sqrt(_)
                    | crate::Layer::Reciprocal(_)
            )
        })
    })
}

fn root_bounds_cache_key(
    cache: &InputSplitRootBoundsCache,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
) -> Option<RootBoundsCacheKey> {
    if !root_collection_is_restart_deterministic(graph, config) {
        return None;
    }
    Some(RootBoundsCacheKey {
        graph_scope: graph.cut_fold_scope(),
        coverage: Arc::from(graph.crown_ibp_collection_cache_fingerprint(input, engine.is_some())),
        options: Arc::from(root_collection_options_identity(config, engine)?),
        spec: Arc::clone(&cache.spec_identity),
    })
}

/// Exact key for only the deterministic typed reference map. Unlike the
/// whole-result key, this deliberately permits restart-seeded alpha methods
/// and supplement-bearing operators: no alpha or optimizer state is stored in
/// this tier. Eligibility is the same exact typed/Step-1 predicate used by
/// alpha initialization, and every ordinary root returns `None`.
fn typed_reference_map_cache_key(
    cache: &InputSplitRootBoundsCache,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
) -> Option<RootBoundsCacheKey> {
    if !config.use_alpha_crown {
        return None;
    }
    let exec_order = graph.exec_order().ok()?;
    let typed = graph.cgan_complete_crown_ibp_root_eligible(&config.alpha_config, exec_order)
        || graph.cgan_sparse_target_complete_root_eligible(&config.alpha_config, exec_order);
    if !typed {
        return None;
    }
    Some(RootBoundsCacheKey {
        graph_scope: graph.cut_fold_scope(),
        coverage: Arc::from(graph.crown_ibp_collection_cache_fingerprint(input, engine.is_some())),
        options: Arc::from(root_collection_options_identity(config, engine)?),
        spec: Arc::clone(&cache.spec_identity),
    })
}

/// Collect the input-split ROOT node-bound map.
///
/// # CONTRACT FOR ANY FUTURE ARM THAT RETURNS `Some(map)` (#root-map-two-guards)
///
/// Every arm below currently returns `None` for the map. That is load-bearing:
/// publishing a root-scope node-bound map is NOT verdict-neutral, and the two
/// conditions that make it safe are NOT implied by each other. Establish BOTH
/// before returning `Some`, or you will silently turn `unsat` into `unknown`.
///
/// 1. `config.input_split_ibp_enhancement` — the SCALAR lane only ever MERGES a
///    supplied root map with the per-domain bounds. With the flag off it treats
///    the map as a full OVERRIDE instead.
///
/// 2. `config.input_split_stacked_rebound` — the BATCHED dense-spec deferred
///    rebound clones a supplied root map VERBATIM for every empty-history
///    domain, and only merges per-domain refinements when this flag is also
///    set. Publishing a root map with `stacked_rebound == false` therefore
///    REPLACES each subdomain's own (tighter) CROWN-IBP intermediates with a
///    root-scope map that was computed over the whole input box.
///
/// Discovered the hard way: a root LP/OBBT tightener built against this seam
/// (preserved out-of-tree as tag `wip/lp-root-obbt`) was designed believing
/// guard 1 alone sufficed. It does not. See
/// `docs/RELAXATION_CAPABILITY_BUILD_2026-07-31.md`.
///
/// Note the map is only ever a TIGHTENING when both guards hold; the intersect
/// must additionally be shrink-only and whole-node atomic, so a partially
/// tightened node cannot mix scopes.
pub(crate) fn collect_input_split_root_node_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
    initial_deadline: Option<Instant>,
    mode_label: &str,
    restart_cache: Option<(&InputSplitRootBoundsCache, Option<Instant>)>,
) -> Result<(
    Option<HashMap<String, BoundedTensor>>,
    Option<GraphAlphaState>,
)> {
    let cache_key = restart_cache.and_then(|(cache, overall_deadline)| {
        cache
            .deadline_matches(overall_deadline)
            .then(|| root_bounds_cache_key(cache, graph, input, config, engine))
            .flatten()
            .map(|key| (cache, key))
    });
    let typed_reference_cache_key = cache_key
        .is_none()
        .then(|| {
            restart_cache.and_then(|(cache, overall_deadline)| {
                cache
                    .deadline_matches(overall_deadline)
                    .then(|| typed_reference_map_cache_key(cache, graph, input, config, engine))
                    .flatten()
                    .map(|key| (cache, key))
            })
        })
        .flatten();
    if restart_cache.is_some() && cache_key.is_none() && typed_reference_cache_key.is_none() {
        eprintln!(
            "[restart-root-cache] bypass mode={mode_label}; root collection is not exactly reusable"
        );
    }
    if let Some((cache, key)) = cache_key.as_ref() {
        if let Some(value) = cache.lookup(key) {
            eprintln!(
                "[restart-root-cache] hit mode={mode_label} hit={} nodes={} alpha={} deadline_remaining={:.3}s",
                cache.hits(),
                value.0.as_ref().map_or(0, HashMap::len),
                value.1.is_some(),
                initial_deadline
                    .map(|deadline| deadline.saturating_duration_since(Instant::now()).as_secs_f64())
                    .unwrap_or(-1.0),
            );
            return Ok(value);
        }
        let collection = cache.note_collection();
        eprintln!(
            "[restart-root-cache] miss mode={mode_label} miss={} collection={collection}; collecting",
            cache.misses(),
        );
    }
    let cached_typed_reference = if let Some((cache, key)) = typed_reference_cache_key.as_ref() {
        match cache.lookup_typed_reference(key) {
            Some(reference) => {
                eprintln!(
                    "[restart-root-cache] typed-map hit mode={mode_label} hit={} nodes={} \
                     deadline_remaining={:.3}s; rebuilding alpha state",
                    cache.typed_reference_hits.load(Ordering::Relaxed),
                    reference.bounds.len(),
                    initial_deadline
                        .map(|deadline| deadline
                            .saturating_duration_since(Instant::now())
                            .as_secs_f64())
                        .unwrap_or(-1.0),
                );
                Some(reference)
            }
            None => {
                let collection = cache.note_typed_reference_collection();
                eprintln!(
                    "[restart-root-cache] typed-map miss mode={mode_label} \
                     collection={collection}; collecting deterministic reference"
                );
                None
            }
        }
    } else {
        None
    };
    if config.use_alpha_crown {
        let mut alpha_config = config.alpha_config.clone();
        alpha_config.deadline = initial_deadline;
        info!("Computing α-CROWN initial bounds for {mode_label}...");
        let precomputed_reference = match cached_typed_reference {
            Some(reference) => Some(reference),
            None if typed_reference_cache_key.is_some() => {
                let exec_order = graph.exec_order()?;
                // This collection was historically inside the DAG-alpha entry,
                // whose L2 guard is disabled before Step 1. Preserve that exact
                // producer contract when lifting only Step 1 into this cache.
                let (bounds, source) = {
                    let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
                    graph.collect_alpha_reference_bounds_with_engine_and_source(
                        input,
                        &alpha_config,
                        engine,
                        exec_order,
                    )?
                };
                let reference = PrecomputedAlphaReferenceBounds { bounds, source };
                if source.is_typed_cgan() {
                    if let Some((cache, key)) = typed_reference_cache_key.as_ref() {
                        cache.store_typed_reference(key.clone(), &reference);
                        eprintln!(
                            "[restart-root-cache] typed-map store mode={mode_label} nodes={} \
                             alpha=false",
                            reference.bounds.len(),
                        );
                    }
                } else {
                    eprintln!(
                        "[restart-root-cache] typed-map declined mode={mode_label}; actual \
                         reference source={source:?}, not caching"
                    );
                }
                Some(reference)
            }
            None => None,
        };
        let (mut bounds, alpha_state) = match precomputed_reference {
            Some(reference) => graph.collect_alpha_crown_bounds_dag_with_engine_and_reference(
                input,
                &alpha_config,
                engine,
                reference,
            )?,
            None => {
                graph.collect_alpha_crown_bounds_dag_with_engine(input, &alpha_config, engine)?
            }
        };
        info!("α-CROWN produced {} intermediate bound sets", bounds.len());
        // Root JOINT per-target intermediate-bound α pass on the INPUT-SPLIT lanes
        // (#root-joint-interm-alpha; this fn is the single choke-point for the
        // disjunctive / single-objective / multi-objective / gpu-bab input-split
        // routes, which bypass multi_objective/root.rs entirely). Same gate/knobs
        // as the graph-lane block; default-OFF => byte-identical. Runs BEFORE the
        // map is frozen by the callers' borrows, so every consumer (root objective
        // check, per-domain fallback, deferred rebound, multineuron inject)
        // inherits the tightened map. SOUND: shrink-only sound-fold intersect in
        // the driver; fail-closed on unsupported topology (prep refusal => the
        // reference bounds are kept unchanged).
        if std::env::var("NY_ROOT_JOINT_INTERM_ALPHA").ok().as_deref() == Some("1") {
            let now = Instant::now();
            // #root-joint-diag (dark; only under the gate): unconditional trace so a
            // probe run can see the seam is REACHED and why it may no-op — the
            // vnncomp tracing filter swallows info!, so this is eprintln!.
            eprintln!(
                "[root-joint-interm-alpha] input-split seam reached ({mode_label}); bounds={} deadline_remaining={:.1}s",
                bounds.len(),
                initial_deadline
                    .map(|d| d.saturating_duration_since(now).as_secs_f32())
                    .unwrap_or(-1.0)
            );
            // #generator-enclosure diag: per-node mean width profile of the frozen
            // map (exec order), to locate WHERE the enclosure blows up.
            if std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_PROBE")
                .ok()
                .as_deref()
                == Some("1")
            {
                if let Ok(order) = graph.exec_order() {
                    for name in order {
                        if let Some(bt) = bounds.get(name) {
                            let n = bt.lower().len().max(1);
                            let w: f32 = bt
                                .lower()
                                .iter()
                                .zip(bt.upper().iter())
                                .map(|(&l, &u)| u - l)
                                .sum::<f32>()
                                / n as f32;
                            eprintln!("[bounds-profile] {name} dim={n} meanw={w:.4}");
                        }
                    }
                }
            }
            let grace = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(30);
            let remaining = initial_deadline
                .map(|d| d.saturating_duration_since(now))
                .unwrap_or_else(|| std::time::Duration::from_secs(grace));
            // The alpha-CROWN collect above CONSUMES the initial-bounds phase
            // deadline (measured on cgan: remaining == 0.0s at this point on
            // every route), so budgeting the joint pass off that spent deadline
            // permanently no-ops it. This is a DARK research lever: when the
            // phase window is exhausted, borrow the grace cap outright —
            // budget-only (sound), bounded by _SECS, and unreachable unless the
            // gate is explicitly set. A scored enable must re-budget properly.
            // Dark research lever: ALWAYS take the grace, ignoring the (already
            // mostly-spent, run-to-run noisy) initial-bounds phase window.
            // Measured: remaining oscillates 0-36s across invocations, and
            // slice=remaining*0.5 starved every walk (v5: 17.2s envelope, all
            // five targets reference-filled); only the grace-borrow path ever
            // succeeded. Budget-borrow is sound (schedule-only) and the lever is
            // unreachable unless the gate is explicitly set; a scored enable
            // must integrate with the real phase budget.
            let _ = remaining;
            let slice = std::time::Duration::from_secs(grace);
            if slice >= std::time::Duration::from_secs(2) {
                // #image-node-crown: fix width-outlier nodes FIRST (the generator's
                // image node never gets a CROWN bound — activation-input targets
                // only — and its one-interval-step entry blows up 100x+), so the
                // joint pass below inherits the tighter stop box.
                // Full slice: measured on cgan, the five amplifier walks +
                // iterated passes need ~1-3 min; a 60s sub-cap starved them to a
                // single target per invocation (ConvT_13 26.25 -> 0.02 proved the
                // mechanism; the cascade needs the rest).
                let n_outlier = graph.tighten_outlier_node_bounds(
                    input,
                    &alpha_state,
                    engine,
                    Some(now + slice),
                    &mut bounds,
                );
                if n_outlier > 0 {
                    eprintln!(
                        "[image-node-crown] input-split root: {n_outlier} outlier node(s) tightened"
                    );
                }
                let targets = crate::beta_crown::engine::graph::propagation::batched::interm_refine::scoped_joint_alpha_targets(graph, &bounds);
                eprintln!(
                    "[root-joint-interm-alpha] input-split targets={} slice_ok=1",
                    targets.len()
                );
                if !targets.is_empty() {
                    let iters = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_ITERS")
                        .ok()
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .unwrap_or(100);
                    let lr = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_LR")
                        .ok()
                        .and_then(|s| s.trim().parse::<f32>().ok())
                        .unwrap_or(0.1);
                    let n = crate::beta_crown::engine::graph::propagation::batched::interm_refine::root_joint_tighten_relu_preactivations(
                        graph,
                        input,
                        &targets,
                        engine,
                        Some(now + slice),
                        iters,
                        lr,
                        &mut bounds,
                    );
                    info!(
                        "[root-joint-interm-alpha] input-split root: {n}/{} target(s) tightened",
                        targets.len()
                    );
                }
            }
        }
        let value = (Some(bounds), Some(alpha_state));
        if let Some((cache, key)) = cache_key {
            cache.store(key, &value);
            eprintln!(
                "[restart-root-cache] store mode={mode_label} nodes={} alpha=true",
                value.0.as_ref().map_or(0, HashMap::len),
            );
        }
        Ok(value)
    } else if config.use_forward_bounds {
        info!("Computing forward-linear initial bounds for {mode_label}...");
        match graph.collect_forward_linear_bounds_dag_with_engine_and_deadline(
            input,
            engine,
            initial_deadline,
        ) {
            Ok(bounds) => {
                info!(
                    "forward-linear produced {} intermediate bound sets",
                    bounds.len()
                );
                let value = (Some(bounds), None);
                if let Some((cache, key)) = cache_key {
                    cache.store(key, &value);
                    eprintln!(
                        "[restart-root-cache] store mode={mode_label} nodes={} alpha=false",
                        value.0.as_ref().map_or(0, HashMap::len),
                    );
                }
                Ok(value)
            }
            Err(NyError::DeadlineExceeded(_)) => {
                info!(
                    "Skipping forward-linear initial bounds for {mode_label} because the warmup deadline expired"
                );
                Ok((None, None))
            }
            Err(NyError::ShapeMismatch {
                ref expected,
                ref got,
            }) => {
                info!(
                    "Skipping forward-linear initial bounds for {mode_label}: \
                     shape mismatch (expected {expected:?}, got {got:?}) — \
                     DAG topology may differ between IBP and CROWN passes"
                );
                Ok((None, None))
            }
            Err(NyError::UnsupportedOp(ref op)) => {
                info!(
                    "Skipping forward-linear initial bounds for {mode_label}: \
                     unsupported op '{op}' in forward-linear path"
                );
                Ok((None, None))
            }
            Err(NyError::UnsupportedConfiguration(ref reason)) => {
                info!(
                    "Skipping forward-linear initial bounds for {mode_label}: \
                     unsupported configuration '{reason}'"
                );
                Ok((None, None))
            }
            Err(err) => Err(err),
        }
    } else {
        Ok((None, None))
    }
}

#[cfg(test)]
#[path = "root_bounds_tests.rs"]
mod tests;
