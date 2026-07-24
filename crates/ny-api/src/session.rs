// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Incremental / batch verification session (P10).
//!
//! A [`VerifierSession`] loads a [`GraphNetwork`] **once** and verifies many
//! [`VerificationSpec`]s against it, reusing the owned network, propagation
//! configuration, and (optional) GEMM engine across every query. This is the
//! core efficiency win over constructing a fresh [`Verifier`] (and re-handing it
//! the network) per call, and is the natural shape for sweep / batch / repeated
//! property workloads.
//!
//! ## Caching (opt-in, conservative, sound)
//!
//! The session carries an opt-in verdict cache. A cached verdict is returned
//! **only** when an incoming spec is byte-for-byte identical (input region,
//! output bounds, input shape, timeout, and every output constraint) to one
//! previously verified *against this same loaded network*. The cache key folds
//! in a structural fingerprint of the owned network, and the session owns its
//! network immutably for its whole lifetime, so a cached verdict can never be
//! returned for a different input region or a different network. Differing input
//! regions hash to different keys and are never confused.
//!
//! The cache stores the [`VerificationResult`] verbatim, including its soundness
//! provenance and any populated proof channel, so a cache hit is the *same*
//! verdict the verifier would have produced — never a stronger claim. Errors are
//! never cached.
//!
//! ```rust,no_run
//! # #[cfg(feature = "propagate")]
//! # fn run() {
//! use ny_api::graph::GraphNetwork;
//! use ny_api::session::VerifierSession;
//! use ny_core::VerificationSpec;
//! # let net: GraphNetwork = unimplemented!();
//! # let specs: Vec<VerificationSpec> = unimplemented!();
//! let mut session = VerifierSession::new(net);
//! let results = session.verify_many(&specs);
//! println!("{} cache hits", session.stats().cache_hits);
//! # let _ = results;
//! # }
//! ```
//!
//! ## Follow-on
//!
//! Warm-starting α/β optimizer state across consecutive specs (so a later, more
//! constrained spec reuses a previous spec's relaxation parameters) is a
//! documented follow-on: the relevant α/β state is not exposed through the
//! `ny_propagate` public surface, so the session reuses the
//! network/config/engine but recomputes bounds per distinct spec rather than
//! warm-starting.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ny_core::{GemmEngine, Result, VerificationResult, VerificationSpec};

use crate::graph::GraphNetwork;
use crate::verify::{PropagationConfig, Verifier};

/// Per-session counters describing how a batch ran.
///
/// Counters are cumulative across every [`VerifierSession::verify`] /
/// [`VerifierSession::verify_many`] call and are reset only by
/// [`VerifierSession::reset_stats`] (clearing the cache via
/// [`VerifierSession::clear_cache`] does **not** reset stats).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionStats {
    /// Total verification queries handled (including cache hits).
    pub queries: usize,
    /// Queries whose verdict was `Verified`.
    pub verified: usize,
    /// Queries served from the verdict cache instead of recomputed.
    pub cache_hits: usize,
    /// Queries that ran the verifier (cache miss; includes ones that errored).
    pub cache_misses: usize,
    /// Queries that returned an `Err` from the verifier.
    pub errors: usize,
}

/// A loaded-once, verify-many verification session.
///
/// Owns a [`GraphNetwork`] plus the [`Verifier`] (propagation config + optional
/// engine) used to verify against it. The network is never mutated after
/// construction, which is what makes the verdict cache sound: the network
/// identity component of every cache key is fixed for the session's lifetime.
pub struct VerifierSession {
    net: GraphNetwork,
    verifier: Verifier,
    /// Structural fingerprint of `net`, folded into every cache key so a cached
    /// verdict can only ever match against the network it was produced on.
    net_fingerprint: u64,
    /// Opt-in verdict cache: spec-key -> verdict. Populated only when
    /// `caching_enabled`.
    cache: HashMap<u64, VerificationResult>,
    /// Whether the verdict cache is consulted/populated.
    caching_enabled: bool,
    stats: SessionStats,
}

impl VerifierSession {
    /// Create a session over `net` using the default propagation config and no
    /// custom GEMM engine. Caching is enabled by default.
    pub fn new(net: GraphNetwork) -> Self {
        Self::with_config(net, PropagationConfig::default())
    }

    /// Create a session over `net` with an explicit propagation config (no
    /// custom GEMM engine). Caching is enabled by default.
    pub fn with_config(net: GraphNetwork, config: PropagationConfig) -> Self {
        let net_fingerprint = network_fingerprint(&net);
        Self {
            net,
            verifier: Verifier::new(config),
            net_fingerprint,
            cache: HashMap::new(),
            caching_enabled: true,
            stats: SessionStats::default(),
        }
    }

    /// Create a session over `net` with an explicit propagation config and a
    /// custom GEMM engine (mirrors [`Verifier::new_with_engine`]). Caching is
    /// enabled by default.
    pub fn with_engine(
        net: GraphNetwork,
        config: PropagationConfig,
        engine: Arc<dyn GemmEngine>,
    ) -> Self {
        let net_fingerprint = network_fingerprint(&net);
        Self {
            net,
            verifier: Verifier::new_with_engine(config, engine),
            net_fingerprint,
            cache: HashMap::new(),
            caching_enabled: true,
            stats: SessionStats::default(),
        }
    }

    /// Enable or disable the verdict cache.
    ///
    /// Disabling does not drop already-cached entries (so re-enabling resumes
    /// hitting them); call [`Self::clear_cache`] to drop them.
    pub fn set_caching_enabled(&mut self, enabled: bool) {
        self.caching_enabled = enabled;
    }

    /// Whether the verdict cache is currently consulted/populated.
    pub fn caching_enabled(&self) -> bool {
        self.caching_enabled
    }

    /// Read-only access to the loaded network.
    pub fn network(&self) -> &GraphNetwork {
        &self.net
    }

    /// Number of distinct verdicts currently held in the cache.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Drop every cached verdict. Cumulative [`SessionStats`] are preserved.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Reset cumulative [`SessionStats`] to zero. Does not touch the cache.
    pub fn reset_stats(&mut self) {
        self.stats = SessionStats::default();
    }

    /// Cumulative statistics for this session.
    pub fn stats(&self) -> &SessionStats {
        &self.stats
    }

    /// Verify the loaded network against one spec.
    ///
    /// On a cache hit (caching enabled and an identical spec was previously
    /// verified against this network) the stored verdict is cloned and returned
    /// without re-running propagation. Otherwise the spec is verified via
    /// [`Verifier::verify_graph`] and, when caching is enabled and the verifier
    /// returns `Ok`, the verdict is cached.
    ///
    /// Errors are never cached (a transient resource/timeout error should not
    /// pin a permanent failure).
    pub fn verify(&mut self, spec: &VerificationSpec) -> Result<VerificationResult> {
        self.stats.queries += 1;

        if self.caching_enabled {
            let key = self.cache_key(spec);
            if let Some(cached) = self.cache.get(&key) {
                self.stats.cache_hits += 1;
                if cached.is_verified() {
                    self.stats.verified += 1;
                }
                return Ok(cached.clone());
            }
            // Cache miss: compute, then store on success.
            self.stats.cache_misses += 1;
            let result = self.verifier.verify_graph(&self.net, spec);
            self.record_outcome(&result);
            if let Ok(ref verdict) = result {
                self.cache.insert(key, verdict.clone());
            }
            result
        } else {
            self.stats.cache_misses += 1;
            let result = self.verifier.verify_graph(&self.net, spec);
            self.record_outcome(&result);
            result
        }
    }

    /// Verify the loaded network against many specs, reusing the network,
    /// propagation config, and engine across every query.
    ///
    /// Results are returned positionally (`out[i]` is the verdict for
    /// `specs[i]`). Identical specs within the batch benefit from the cache: the
    /// first occurrence runs the verifier, later occurrences are cache hits with
    /// the same verdict.
    ///
    /// This intentionally loops on the shared session rather than calling
    /// `ny_propagate::verify_parallel`: that helper parallelizes *one* spec
    /// across sequence positions, whereas this method's win is sharing the
    /// loaded network/config/engine and the verdict cache across *many distinct
    /// specs*, which a per-spec parallel helper would not provide.
    pub fn verify_many(&mut self, specs: &[VerificationSpec]) -> Vec<Result<VerificationResult>> {
        specs.iter().map(|spec| self.verify(spec)).collect()
    }

    /// Update verified/error counters from a freshly computed result.
    fn record_outcome(&mut self, result: &Result<VerificationResult>) {
        match result {
            Ok(verdict) => {
                if verdict.is_verified() {
                    self.stats.verified += 1;
                }
            }
            Err(_) => self.stats.errors += 1,
        }
    }

    /// Deterministic cache key over (network fingerprint, full spec content).
    ///
    /// Folding in `net_fingerprint` guarantees a verdict produced on one network
    /// can never be served for a query against a different network.
    fn cache_key(&self, spec: &VerificationSpec) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.net_fingerprint.hash(&mut hasher);
        hash_spec(spec, &mut hasher);
        hasher.finish()
    }
}

/// Structural fingerprint of a network: node count, ordered node names, and the
/// output node name.
///
/// The owned network is immutable for the session's lifetime, so this fixed
/// fingerprint fully scopes the cache to "this exact network". It is a
/// fingerprint, not a content hash of weights, because the session can never
/// observe two structurally identical networks with different weights (it owns
/// exactly one network and never mutates it); the fingerprint exists to make the
/// network-identity component of the key explicit and collision-resistant.
fn network_fingerprint(net: &GraphNetwork) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    net.num_nodes().hash(&mut hasher);
    for name in net.node_names() {
        name.hash(&mut hasher);
    }
    net.output_name().hash(&mut hasher);
    hasher.finish()
}

/// Fold every soundness-relevant field of a spec into `hasher`.
///
/// Floats are hashed by their IEEE-754 bit pattern (`to_bits`), which is total
/// and deterministic. Input bounds are guaranteed non-NaN by the spec
/// constructor; `to_bits` is well-defined regardless.
fn hash_spec(spec: &VerificationSpec, hasher: &mut impl Hasher) {
    hash_bounds(spec.input_bounds(), hasher);
    hash_bounds(spec.output_bounds(), hasher);

    match spec.input_shape() {
        Some(shape) => {
            1u8.hash(hasher);
            shape.hash(hasher);
        }
        None => 0u8.hash(hasher),
    }

    spec.timeout_ms().hash(hasher);

    spec.output_constraints().len().hash(hasher);
    for c in spec.output_constraints() {
        hash_output_constraint(c, hasher);
    }
}

fn hash_bounds(bounds: &[ny_core::Bound], hasher: &mut impl Hasher) {
    bounds.len().hash(hasher);
    for b in bounds {
        b.lower().to_bits().hash(hasher);
        b.upper().to_bits().hash(hasher);
    }
}

fn hash_output_constraint(c: &ny_core::OutputConstraint, hasher: &mut impl Hasher) {
    use ny_core::{ConstraintKind, OutputConstraint};
    match c {
        OutputConstraint::Bounds(bounds) => {
            0u8.hash(hasher);
            hash_bounds(bounds, hasher);
        }
        OutputConstraint::Linear { coeffs, bias, kind } => {
            1u8.hash(hasher);
            coeffs.len().hash(hasher);
            for coeff in coeffs {
                coeff.to_bits().hash(hasher);
            }
            bias.to_bits().hash(hasher);
            match kind {
                ConstraintKind::Le => 0u8.hash(hasher),
                ConstraintKind::Ge => 1u8.hash(hasher),
            }
        }
        OutputConstraint::ArgmaxMargin { class } => {
            2u8.hash(hasher);
            class.hash(hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphNetwork, GraphNode};
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use ndarray::{Array1, Array2};
    use ny_core::{Bound, VerificationSpec};

    /// Tiny 1->1 identity linear graph: out = x (weight 1, bias 0).
    fn identity_graph() -> GraphNetwork {
        let w = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("1x1 weight");
        let linear = LinearLayer::new(w, Some(Array1::zeros(1))).expect("valid linear layer");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.set_output("linear");
        graph
    }

    /// Tiny linear+ReLU+linear 2->2 graph (a structurally distinct network).
    fn relu_graph() -> GraphNetwork {
        let w1 = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("2x2 w1");
        let linear1 = LinearLayer::new(w1, Some(Array1::zeros(2))).expect("valid linear1");
        let w2 = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("2x2 w2");
        let linear2 = LinearLayer::new(w2, Some(Array1::zeros(2))).expect("valid linear2");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");
        graph
    }

    /// Loose spec: input [0,1] -> output [0,1], allowed [-1,2]: trivially verified.
    fn loose_spec() -> VerificationSpec {
        VerificationSpec::new(vec![Bound::new(0.0, 1.0)], vec![Bound::new(-1.0, 2.0)])
            .expect("valid spec")
    }

    /// A different input region than `loose_spec` (so it must hash differently).
    fn other_region_spec() -> VerificationSpec {
        VerificationSpec::new(vec![Bound::new(2.0, 3.0)], vec![Bound::new(-10.0, 10.0)])
            .expect("valid spec")
    }

    #[test]
    fn session_verify_matches_single_shot() {
        let graph = identity_graph();
        let spec = loose_spec();

        // Single-shot baseline via a fresh verifier.
        let baseline = Verifier::new(PropagationConfig::default())
            .verify_graph(&graph, &spec)
            .expect("baseline verify");

        let mut session = VerifierSession::new(identity_graph());
        let got = session.verify(&spec).expect("session verify");

        assert_eq!(
            baseline.is_verified(),
            got.is_verified(),
            "session verdict must match single-shot verify"
        );
        assert!(got.is_verified(), "loose spec must verify");
        assert_eq!(session.stats().queries, 1);
        assert_eq!(session.stats().verified, 1);
        assert_eq!(session.stats().cache_hits, 0);
        assert_eq!(session.stats().cache_misses, 1);
    }

    #[test]
    fn identical_repeated_spec_is_cache_hit_with_same_verdict() {
        let mut session = VerifierSession::new(identity_graph());
        let spec = loose_spec();

        let first = session.verify(&spec).expect("first verify");
        assert_eq!(session.stats().cache_hits, 0, "first is a miss");
        assert_eq!(session.stats().cache_misses, 1);

        let second = session.verify(&spec).expect("second verify");
        assert_eq!(
            session.stats().cache_hits,
            1,
            "identical repeated spec must be a cache hit"
        );
        // Cache miss count must NOT increase on a hit.
        assert_eq!(session.stats().cache_misses, 1);

        // SAME verdict on the cache hit.
        assert_eq!(first.is_verified(), second.is_verified());
        assert!(second.is_verified());
        assert_eq!(session.cache_len(), 1);
    }

    #[test]
    fn verify_many_matches_single_shot_and_caches_repeats() {
        let baseline_net = identity_graph();
        let baseline_verifier = Verifier::new(PropagationConfig::default());

        let s0 = loose_spec();
        let s1 = other_region_spec();
        // The third spec is identical to the first -> should be a cache hit.
        let specs = vec![loose_spec(), other_region_spec(), loose_spec()];

        let expect0 = baseline_verifier
            .verify_graph(&baseline_net, &s0)
            .expect("baseline s0");
        let expect1 = baseline_verifier
            .verify_graph(&baseline_net, &s1)
            .expect("baseline s1");

        let mut session = VerifierSession::new(identity_graph());
        let results = session.verify_many(&specs);

        assert_eq!(results.len(), 3);
        let r0 = results[0].as_ref().expect("r0 ok");
        let r1 = results[1].as_ref().expect("r1 ok");
        let r2 = results[2].as_ref().expect("r2 ok");

        assert_eq!(r0.is_verified(), expect0.is_verified());
        assert_eq!(r1.is_verified(), expect1.is_verified());
        // The repeated spec yields the SAME verdict as its first occurrence.
        assert_eq!(r2.is_verified(), r0.is_verified());

        assert_eq!(session.stats().queries, 3);
        assert_eq!(
            session.stats().cache_hits,
            1,
            "exactly the repeated spec should hit"
        );
        // Two distinct specs -> two misses; the repeat is a hit.
        assert_eq!(session.stats().cache_misses, 2);
        assert_eq!(session.cache_len(), 2, "two distinct specs cached");
    }

    #[test]
    fn different_input_regions_are_not_confused() {
        let mut session = VerifierSession::new(identity_graph());

        let a = loose_spec(); // input [0,1]
        let b = other_region_spec(); // input [2,3] — different region

        let _ = session.verify(&a).expect("verify a");
        let _ = session.verify(&b).expect("verify b");

        // Two distinct regions must both be misses (no false cache hit).
        assert_eq!(
            session.stats().cache_hits,
            0,
            "different input regions must not produce a cache hit"
        );
        assert_eq!(session.stats().cache_misses, 2);
        assert_eq!(session.cache_len(), 2);

        // Their cache keys must differ.
        assert_ne!(
            session.cache_key(&a),
            session.cache_key(&b),
            "distinct input regions must hash to distinct keys"
        );
    }

    #[test]
    fn distinct_networks_have_distinct_fingerprints() {
        // Sanity: structurally different networks fingerprint differently, so a
        // verdict cached on one network could never be served for another.
        assert_ne!(
            network_fingerprint(&identity_graph()),
            network_fingerprint(&relu_graph()),
            "structurally distinct networks must fingerprint differently"
        );
    }

    #[test]
    fn clear_cache_drops_entries_but_keeps_stats() {
        let mut session = VerifierSession::new(identity_graph());
        let spec = loose_spec();
        let _ = session.verify(&spec).expect("verify");
        assert_eq!(session.cache_len(), 1);

        session.clear_cache();
        assert_eq!(session.cache_len(), 0, "clear_cache drops entries");
        assert_eq!(session.stats().queries, 1, "stats are preserved");

        // After clearing, the same spec is recomputed (a fresh miss), not a hit.
        let _ = session.verify(&spec).expect("verify again");
        assert_eq!(session.stats().cache_hits, 0, "cleared cache yields a miss");
        assert_eq!(session.stats().cache_misses, 2);
    }

    #[test]
    fn caching_disabled_never_hits() {
        let mut session = VerifierSession::new(identity_graph());
        session.set_caching_enabled(false);
        assert!(!session.caching_enabled());

        let spec = loose_spec();
        let _ = session.verify(&spec).expect("verify 1");
        let _ = session.verify(&spec).expect("verify 2");

        assert_eq!(session.stats().cache_hits, 0, "caching disabled: no hits");
        assert_eq!(session.stats().cache_misses, 2);
        assert_eq!(session.cache_len(), 0, "nothing cached when disabled");
    }

    #[test]
    fn with_config_is_honored() {
        use crate::verify::PropagationMethod;
        let cfg = PropagationConfig {
            method: PropagationMethod::Crown,
            ..Default::default()
        };
        let mut session = VerifierSession::with_config(identity_graph(), cfg);
        let got = session.verify(&loose_spec()).expect("verify with crown");
        assert!(got.is_verified());
    }
}
