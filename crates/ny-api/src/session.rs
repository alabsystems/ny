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
//! ## Warm start (opt-in, off by default)
//!
//! The expensive half of a query is bound propagation, not the property check.
//! For every propagation method the session can run — `Ibp`, `Crown`,
//! `AlphaCrown`, `SdpCrown` — `Verifier::verify_graph` propagates from
//! `(network, input region, input shape, timeout)` ALONE and only then compares
//! the resulting enclosure against `spec.output_bounds()`. The α-CROWN slope
//! optimization is part of that property-independent half: it tightens the
//! network's output enclosure, not a per-property margin.
//!
//! [`VerifierSession::set_warm_start_enabled`] turns that observation into
//! reuse. The session keeps, per input region it has propagated, the computed
//! enclosure a `Verified` verdict carried, together with that run's soundness
//! provenance and method label. A later spec over the SAME region (same input
//! bounds bit-for-bit, same input shape, same timeout) whose required output
//! bounds already CONTAIN that enclosure is answered `Verified` without running
//! propagation again — the optimized bounds are reused directly, which is the
//! result α-state warm-starting is trying to approximate, obtained exactly.
//!
//! Soundness, stated precisely:
//!
//! * the stored enclosure is a real, previously-computed sound enclosure of the
//!   network's outputs over that region, so containment in the new spec's
//!   required bounds proves the new property;
//! * the reuse condition (`computed[i].lower() >= required[i].lower() &&
//!   computed[i].upper() <= required[i].upper()`, over at least as many
//!   computed bounds as required ones) is STRICTLY STRONGER than the check the
//!   verifier itself performs — with a NaN anywhere it is `false` while the
//!   verifier's negated form would pass — so the session synthesizes `Verified`
//!   only where the verifier would also have said `Verified`, and any other
//!   outcome falls through to a full verify;
//! * only `Verified` is ever synthesized: `Unknown` / `Violated` / `Timeout`
//!   always come from the verifier itself;
//! * region identity is re-checked field-by-field on the stored snapshot, not
//!   left to the hash, so a key collision cannot serve bounds from a different
//!   region;
//! * `BetaCrown` is excluded — its branch-and-bound is property-driven, so its
//!   propagation is NOT a function of the region alone;
//! * specs carrying `output_constraints` are excluded on both sides (store and
//!   hit), since this path's verdict is a statement about `output_bounds`.
//!
//! ## What is NOT wired, and why
//!
//! True α/β **optimizer-state** warm starting — carrying a previous spec's
//! slopes into the next optimization — remains unreachable from this facade, so
//! the audit note claiming the machinery is merely unused is wrong on the
//! reachability question, not on the desirability one. `ny_propagate` does
//! expose a warm-start entry (`GraphNetwork::collect_alpha_crown_bounds_dag_warm`)
//! and root-exports `AlphaState`, but the entry's `warm_start` parameter is a
//! `GraphAlphaState`, which is `pub(crate)` in `ny-propagate`; and `Verifier`
//! has no method that accepts an alpha state for a `verify_graph` call. There
//! is therefore no publicly nameable way to capture the state one call produced
//! and hand it to the next.
//!
//! The validity conditions such a wiring would have to honour are worth
//! recording, because they are what makes the reuse above the *sound* subset.
//! ny's own BaB warm start (`streaming` of alpha across domains) does NOT copy
//! a parent's state wholesale: it recomputes the sub-domain's reference bounds,
//! builds a FRESH alpha state so the unstable masks and per-node widths match
//! the new domain, and only then seeds the slope VALUES from the parent
//! (clamped into their valid range). Slopes are free parameters of a valid
//! relaxation, so seeded values change tightness only; the masks and
//! pre-activation bounds are what carry soundness, and they must be rebuilt for
//! the new input region. A cross-spec wiring on the SAME region would reuse
//! both legitimately — but on the same region the propagation result is
//! identical anyway, which is exactly what the bound reuse above exploits
//! without needing any of the private state.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ny_core::{
    Bound, GemmEngine, MethodUsed, Result, SoundnessProvenance, VerificationResult,
    VerificationSpec,
};

use crate::graph::GraphNetwork;
use crate::verify::{PropagationConfig, PropagationMethod, Verifier};

/// Per-session counters describing how a batch ran.
///
/// Counters are cumulative across every [`VerifierSession::verify`] /
/// [`VerifierSession::verify_many`] call and are reset only by
/// [`VerifierSession::reset_stats`] (clearing the cache via
/// [`VerifierSession::clear_cache`] does **not** reset stats).
///
/// `queries == cache_hits + warm_start_hits + cache_misses`: every query is
/// served from the verdict cache, from a reused propagation, or by running the
/// verifier.
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
    /// Queries answered by reusing a previous spec's propagated enclosure over
    /// the same input region (see the module's warm-start section). Always `0`
    /// unless [`VerifierSession::set_warm_start_enabled`] was called. A
    /// warm-start hit runs neither the verifier nor the verdict cache, so it
    /// increments neither `cache_hits` nor `cache_misses`.
    pub warm_start_hits: usize,
}

/// One completed, property-independent propagation over a single input region,
/// harvested from a `Verified` verdict.
///
/// The region-identity fields are kept VERBATIM (not just hashed) so a hash
/// collision can never serve an enclosure computed over a different region: a
/// candidate hit is accepted only after [`Self::matches_region`] confirms the
/// bit patterns, shape, and timeout all agree.
#[derive(Debug, Clone)]
struct PropagationSnapshot {
    /// The input box this enclosure was propagated over.
    input_bounds: Vec<Bound>,
    /// The spec's declared input shape (`None` = flat).
    input_shape: Option<Vec<usize>>,
    /// The spec's timeout, which feeds the propagation deadline and can change
    /// how far the cascade degrades — so it is part of the region's identity.
    timeout_ms: Option<u64>,
    /// The computed output enclosure, exactly as the `Verified` verdict carried
    /// it (the verifier's own flattened output bounds).
    computed: Vec<Bound>,
    /// The provenance of THIS propagation — reused verbatim, because the
    /// reused verdict rests on exactly these bounds.
    provenance: SoundnessProvenance,
    /// The method label THIS propagation recorded.
    actual_method: Option<MethodUsed>,
}

impl PropagationSnapshot {
    /// Whether `spec` asks about the very region this snapshot was propagated
    /// over. Floats compare by bit pattern: total, and exact.
    fn matches_region(&self, spec: &VerificationSpec) -> bool {
        let incoming = spec.input_bounds();
        self.input_bounds.len() == incoming.len()
            && self.input_bounds.iter().zip(incoming).all(|(a, b)| {
                a.lower().to_bits() == b.lower().to_bits()
                    && a.upper().to_bits() == b.upper().to_bits()
            })
            && self.input_shape.as_deref() == spec.input_shape()
            && self.timeout_ms == spec.timeout_ms()
    }
}

/// Whether a previously computed enclosure `computed` already discharges the
/// `required` output bounds.
///
/// This is deliberately the POSITIVE form of the verifier's own containment
/// test, not the negation of its rejection test. The verifier reports `Unknown`
/// when `computed.lower() < required.lower() || computed.upper() >
/// required.upper()`, so with a NaN endpoint both comparisons are `false` and
/// it would report `Verified`; here the same NaN makes `>=`/`<=` false, this
/// returns `false`, and the query falls through to a real verify. The predicate
/// is therefore strictly stronger than the verifier's: everything it admits the
/// verifier admits too, and nothing it rejects is ever answered from reuse.
///
/// (`Bound`'s constructors reject NaN outright, so on every value that can
/// actually reach here the two agree exactly — the positive form costs nothing
/// and keeps the implication one-directional by construction rather than by
/// relying on that upstream invariant.)
fn computed_bounds_discharge(computed: &[Bound], required: &[Bound]) -> bool {
    // Empty requirements are rejected by the verifier itself (`InvalidSpec`),
    // and truncation would silently skip a requirement.
    if required.is_empty() || computed.len() < required.len() {
        return false;
    }
    computed
        .iter()
        .zip(required)
        .all(|(c, r)| c.lower() >= r.lower() && c.upper() <= r.upper())
}

/// Whether propagation under `method` is a function of the input region alone
/// (and therefore reusable across specs that differ only in their property).
///
/// `BetaCrown` is excluded: its branch-and-bound splits are driven by the
/// property being proven, so two specs over the same region can take entirely
/// different search paths and its verdict is not derived from a
/// property-independent enclosure.
fn method_propagates_independently_of_property(method: PropagationMethod) -> bool {
    match method {
        PropagationMethod::Ibp
        | PropagationMethod::Crown
        | PropagationMethod::AlphaCrown
        | PropagationMethod::SdpCrown => true,
        PropagationMethod::BetaCrown => false,
    }
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
    /// The propagation method this session's verifier runs. Recorded because
    /// warm-start reuse is only sound for the property-independent methods
    /// (see [`method_propagates_independently_of_property`]).
    method: PropagationMethod,
    /// Structural fingerprint of `net`, folded into every cache key so a cached
    /// verdict can only ever match against the network it was produced on.
    net_fingerprint: u64,
    /// Opt-in verdict cache: spec-key -> verdict. Populated only when
    /// `caching_enabled`.
    cache: HashMap<u64, VerificationResult>,
    /// Whether the verdict cache is consulted/populated.
    caching_enabled: bool,
    /// Opt-in warm-start store: region-key -> the tightest enclosure this
    /// session has propagated over that region. Populated only when
    /// `warm_start_enabled`.
    warm_start: HashMap<u64, PropagationSnapshot>,
    /// Whether propagated enclosures are reused across specs. OFF by default:
    /// enabling it changes which queries run propagation, so it is an explicit
    /// caller decision.
    warm_start_enabled: bool,
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
        let method = config.method;
        Self {
            net,
            verifier: Verifier::new(config),
            method,
            net_fingerprint,
            cache: HashMap::new(),
            caching_enabled: true,
            warm_start: HashMap::new(),
            warm_start_enabled: false,
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
        let method = config.method;
        Self {
            net,
            verifier: Verifier::new_with_engine(config, engine),
            method,
            net_fingerprint,
            cache: HashMap::new(),
            caching_enabled: true,
            warm_start: HashMap::new(),
            warm_start_enabled: false,
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

    /// Enable or disable warm starting: reusing a previously propagated output
    /// enclosure to answer a LATER spec over the same input region whose
    /// required bounds already contain it (see the module's warm-start
    /// section).
    ///
    /// OFF by default. Enabling it never changes a verdict — a reused answer is
    /// the verdict the verifier would have produced from the same bounds, and
    /// anything the reuse condition does not admit falls through to a full
    /// verify. It has no effect when the session's propagation method is
    /// `BetaCrown`, whose search is property-driven.
    ///
    /// Disabling does not drop already-stored enclosures (so re-enabling
    /// resumes using them); call [`Self::clear_warm_start`] to drop them.
    pub fn set_warm_start_enabled(&mut self, enabled: bool) {
        self.warm_start_enabled = enabled;
    }

    /// Whether warm starting is currently consulted/populated.
    pub fn warm_start_enabled(&self) -> bool {
        self.warm_start_enabled
    }

    /// Whether warm starting can do anything at all for this session: it is
    /// enabled AND the configured propagation method propagates independently
    /// of the property.
    pub fn warm_start_effective(&self) -> bool {
        self.warm_start_enabled && method_propagates_independently_of_property(self.method)
    }

    /// Number of distinct input regions with a stored propagated enclosure.
    pub fn warm_start_len(&self) -> usize {
        self.warm_start.len()
    }

    /// Drop every stored enclosure. Cumulative [`SessionStats`] are preserved.
    pub fn clear_warm_start(&mut self) {
        self.warm_start.clear();
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
    ///
    /// When warm starting is enabled ([`Self::set_warm_start_enabled`]) and the
    /// verdict cache misses, a previously propagated enclosure over the same
    /// input region is consulted before propagation is re-run; see the module's
    /// warm-start section for the exact (strictly conservative) condition.
    pub fn verify(&mut self, spec: &VerificationSpec) -> Result<VerificationResult> {
        self.stats.queries += 1;

        // 1. Exact verdict cache: an identical spec on this network.
        let cache_key = self.caching_enabled.then(|| self.cache_key(spec));
        if let Some(key) = cache_key {
            if let Some(cached) = self.cache.get(&key) {
                self.stats.cache_hits += 1;
                if cached.is_verified() {
                    self.stats.verified += 1;
                }
                return Ok(cached.clone());
            }
        }

        // 2. Warm start: a DIFFERENT spec over the same input region whose
        // required bounds already contain an enclosure we propagated earlier.
        // Only `Verified` is ever synthesized here; everything else falls
        // through to the verifier below.
        if let Some(verdict) = self.warm_start_verdict(spec) {
            self.stats.warm_start_hits += 1;
            self.stats.verified += 1;
            return Ok(verdict);
        }

        // 3. Run the verifier, then store what it produced.
        self.stats.cache_misses += 1;
        let result = self.verifier.verify_graph(&self.net, spec);
        self.record_outcome(&result);
        if let Ok(ref verdict) = result {
            if let Some(key) = cache_key {
                self.cache.insert(key, verdict.clone());
            }
            self.record_warm_start(spec, verdict);
        }
        result
    }

    /// The reused `Verified` verdict for `spec`, or `None` to run the verifier.
    ///
    /// Every gate here is a refusal to reuse, never a claim: an ineligible
    /// method, a property this path does not speak for, a region that does not
    /// match bit-for-bit, or an enclosure that does not already discharge the
    /// requirement all return `None` and cost only a hash lookup.
    fn warm_start_verdict(&self, spec: &VerificationSpec) -> Option<VerificationResult> {
        if !self.warm_start_effective() {
            return None;
        }
        // This path's verdict is a statement about `output_bounds` only
        // (`verify_graph` does not read `output_constraints`), so a spec that
        // carries constraints is never answered from it.
        if !spec.output_constraints().is_empty() {
            return None;
        }
        let snapshot = self.warm_start.get(&self.region_key(spec))?;
        // Hash-collision guard: confirm the stored region IS this spec's.
        if !snapshot.matches_region(spec) {
            return None;
        }
        if !computed_bounds_discharge(&snapshot.computed, spec.output_bounds()) {
            return None;
        }
        Some(VerificationResult::Verified {
            provenance: snapshot.provenance.clone(),
            output_bounds: snapshot.computed.clone(),
            proof: None,
            actual_method: snapshot.actual_method.clone(),
        })
    }

    /// Record the enclosure a fresh `Verified` verdict carried, keeping the
    /// TIGHTEST enclosure seen for the region (a later run can degrade — e.g.
    /// a deadline-driven CROWN -> IBP fallback — and replacing a tight stored
    /// enclosure with a loose one would only lose future reuse).
    fn record_warm_start(&mut self, spec: &VerificationSpec, verdict: &VerificationResult) {
        if !self.warm_start_effective() || !spec.output_constraints().is_empty() {
            return;
        }
        let VerificationResult::Verified {
            provenance,
            output_bounds,
            actual_method,
            ..
        } = verdict
        else {
            return;
        };
        if output_bounds.is_empty() {
            return;
        }
        let key = self.region_key(spec);
        let snapshot = PropagationSnapshot {
            input_bounds: spec.input_bounds().to_vec(),
            input_shape: spec.input_shape().map(<[usize]>::to_vec),
            timeout_ms: spec.timeout_ms(),
            computed: output_bounds.clone(),
            provenance: provenance.clone(),
            actual_method: actual_method.clone(),
        };
        let replace = match self.warm_start.get(&key) {
            None => true,
            Some(existing) => {
                // A different region behind the same key (a hash collision):
                // take the fresh snapshot, since this key is what the next
                // lookup for THIS region will use. Same region: keep whichever
                // enclosure is tighter.
                !existing.matches_region(spec)
                    || computed_bounds_discharge(&snapshot.computed, &existing.computed)
            }
        };
        if replace {
            self.warm_start.insert(key, snapshot);
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

    /// Deterministic warm-start key over (network fingerprint, input region,
    /// input shape, timeout) — exactly the inputs a property-independent
    /// propagation consumes, and NOTHING from the output property.
    ///
    /// This is a lookup key only: [`PropagationSnapshot::matches_region`]
    /// re-checks the same fields verbatim before any stored enclosure is used,
    /// so the key's collision behaviour cannot affect soundness.
    fn region_key(&self, spec: &VerificationSpec) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.net_fingerprint.hash(&mut hasher);
        hash_bounds(spec.input_bounds(), &mut hasher);
        match spec.input_shape() {
            Some(shape) => {
                1u8.hash(&mut hasher);
                shape.hash(&mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }
        spec.timeout_ms().hash(&mut hasher);
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

fn hash_bounds(bounds: &[Bound], hasher: &mut impl Hasher) {
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

    // ---------------------------------------------------------------------
    // Warm start: reuse of a property-independent propagation across specs.
    // ---------------------------------------------------------------------

    /// Same input region as `loose_spec`, a WIDER required output interval —
    /// a different spec whose property the loose run's enclosure discharges.
    fn wider_output_spec() -> VerificationSpec {
        VerificationSpec::new(vec![Bound::new(0.0, 1.0)], vec![Bound::new(-5.0, 5.0)])
            .expect("valid spec")
    }

    /// Same input region again, but a required interval far tighter than the
    /// identity network's own output range [0, 1] — reuse must NOT fire.
    fn unreachable_output_spec() -> VerificationSpec {
        VerificationSpec::new(vec![Bound::new(0.0, 1.0)], vec![Bound::new(0.4, 0.6)])
            .expect("valid spec")
    }

    fn warm_session() -> VerifierSession {
        let mut session = VerifierSession::new(identity_graph());
        session.set_warm_start_enabled(true);
        session
    }

    #[test]
    fn warm_start_is_off_by_default() {
        let mut session = VerifierSession::new(identity_graph());
        assert!(!session.warm_start_enabled(), "warm start is opt-in");
        assert!(!session.warm_start_effective());

        let _ = session.verify(&loose_spec()).expect("verify a");
        let _ = session.verify(&wider_output_spec()).expect("verify b");

        assert_eq!(
            session.stats().warm_start_hits,
            0,
            "no reuse without opt-in"
        );
        assert_eq!(
            session.stats().cache_misses,
            2,
            "both specs run the verifier when warm start is off"
        );
        assert_eq!(session.warm_start_len(), 0, "nothing stored when off");
    }

    #[test]
    fn warm_start_reuses_propagation_for_a_weaker_property_on_the_same_region() {
        let mut session = warm_session();
        assert!(session.warm_start_effective(), "default method is reusable");

        let first = session.verify(&loose_spec()).expect("first verify");
        assert!(first.is_verified());
        assert_eq!(session.stats().cache_misses, 1, "first ran the verifier");
        assert_eq!(session.warm_start_len(), 1, "one region stored");

        // A DIFFERENT spec (wider required interval) over the same region.
        let second = session.verify(&wider_output_spec()).expect("second verify");
        assert_eq!(
            session.stats().warm_start_hits,
            1,
            "the weaker property must be answered from the stored enclosure"
        );
        assert_eq!(
            session.stats().cache_misses,
            1,
            "the second query must NOT have run the verifier"
        );
        assert!(second.is_verified());

        // DIFFERENTIAL: the reused verdict is the verdict a fresh verifier
        // produces for that same spec — same verdict kind, same bounds,
        // same provenance, same method label.
        let fresh = Verifier::new(PropagationConfig::default())
            .verify_graph(&identity_graph(), &wider_output_spec())
            .expect("fresh verify");
        match (&second, &fresh) {
            (
                VerificationResult::Verified {
                    output_bounds: reused,
                    provenance: reused_prov,
                    actual_method: reused_method,
                    ..
                },
                VerificationResult::Verified {
                    output_bounds: direct,
                    provenance: direct_prov,
                    actual_method: direct_method,
                    ..
                },
            ) => {
                assert_eq!(reused, direct, "reused enclosure matches a fresh run");
                assert_eq!(
                    format!("{reused_prov:?}"),
                    format!("{direct_prov:?}"),
                    "provenance matches"
                );
                assert_eq!(
                    format!("{reused_method:?}"),
                    format!("{direct_method:?}"),
                    "method label matches"
                );
            }
            other => panic!("both must be Verified, got {other:?}"),
        }

        assert_eq!(session.stats().queries, 2);
        assert_eq!(
            session.stats().verified,
            2,
            "both queries verified (one of them from reuse)"
        );
    }

    #[test]
    fn warm_start_falls_through_when_the_enclosure_does_not_discharge() {
        let mut session = warm_session();
        let _ = session.verify(&loose_spec()).expect("seed the region");
        assert_eq!(session.warm_start_len(), 1);

        // Identity net over [0,1] produces bounds [0,1], which do NOT sit
        // inside [0.4, 0.6]: reuse must refuse and the verifier must run.
        let got = session
            .verify(&unreachable_output_spec())
            .expect("verify runs");
        assert_eq!(
            session.stats().warm_start_hits,
            0,
            "a non-discharging requirement must never be answered from reuse"
        );
        assert_eq!(session.stats().cache_misses, 2, "the verifier ran");

        let fresh = Verifier::new(PropagationConfig::default())
            .verify_graph(&identity_graph(), &unreachable_output_spec())
            .expect("fresh verify");
        assert_eq!(
            got.is_verified(),
            fresh.is_verified(),
            "fall-through verdict must be the verifier's own"
        );
        assert!(!got.is_verified(), "[0,1] is not inside [0.4, 0.6]");
    }

    #[test]
    fn warm_start_never_crosses_input_regions() {
        let mut session = warm_session();
        let _ = session.verify(&loose_spec()).expect("region [0,1]");
        // A different region with a very permissive requirement: if regions
        // were confused this WOULD be served from the stored enclosure.
        let _ = session.verify(&other_region_spec()).expect("region [2,3]");

        assert_eq!(
            session.stats().warm_start_hits,
            0,
            "a different input region must never reuse another region's bounds"
        );
        assert_eq!(session.stats().cache_misses, 2);
        assert_eq!(session.warm_start_len(), 2, "two distinct regions stored");
    }

    #[test]
    fn warm_start_ignores_specs_carrying_output_constraints() {
        use ny_core::OutputConstraint;
        let mut session = warm_session();
        let _ = session.verify(&loose_spec()).expect("seed the region");

        // Same region, permissive output_bounds — but the spec also carries a
        // constraint this path's verdict does not speak for.
        let constrained =
            VerificationSpec::new(vec![Bound::new(0.0, 1.0)], vec![Bound::new(-5.0, 5.0)])
                .expect("valid spec")
                .with_output_constraints(vec![OutputConstraint::Bounds(vec![Bound::new(
                    -5.0, 5.0,
                )])])
                .expect("valid constraints");

        let _ = session.verify(&constrained).expect("verify runs");
        assert_eq!(
            session.stats().warm_start_hits,
            0,
            "a constrained spec must not be answered from an output_bounds enclosure"
        );
        assert_eq!(session.stats().cache_misses, 2, "the verifier ran");
    }

    #[test]
    fn warm_start_is_inert_for_property_driven_beta_crown() {
        use crate::verify::PropagationMethod;
        let cfg = PropagationConfig {
            method: PropagationMethod::BetaCrown,
            ..Default::default()
        };
        let mut session = VerifierSession::with_config(identity_graph(), cfg);
        session.set_warm_start_enabled(true);
        assert!(session.warm_start_enabled(), "the flag is set");
        assert!(
            !session.warm_start_effective(),
            "beta-CROWN's search is property-driven — reuse must stay inert"
        );
        // Every other method is reusable.
        for method in [
            PropagationMethod::Ibp,
            PropagationMethod::Crown,
            PropagationMethod::AlphaCrown,
            PropagationMethod::SdpCrown,
        ] {
            assert!(
                method_propagates_independently_of_property(method),
                "{method:?} propagates from the region alone"
            );
        }
        assert!(!method_propagates_independently_of_property(
            PropagationMethod::BetaCrown
        ));
    }

    #[test]
    fn clear_warm_start_drops_entries_but_keeps_stats() {
        let mut session = warm_session();
        let _ = session.verify(&loose_spec()).expect("seed");
        assert_eq!(session.warm_start_len(), 1);

        session.clear_warm_start();
        assert_eq!(session.warm_start_len(), 0);
        assert_eq!(session.stats().queries, 1, "stats preserved");

        // After clearing, the weaker property recomputes instead of reusing.
        let _ = session.verify(&wider_output_spec()).expect("verify");
        assert_eq!(session.stats().warm_start_hits, 0);
        assert_eq!(session.stats().cache_misses, 2);
    }

    #[test]
    fn warm_start_and_verdict_cache_stay_independent() {
        let mut session = warm_session();
        let spec = loose_spec();
        let _ = session.verify(&spec).expect("first");
        // An IDENTICAL spec is still a verdict-cache hit, not a warm-start hit:
        // the cache is consulted first.
        let _ = session.verify(&spec).expect("second");
        assert_eq!(session.stats().cache_hits, 1);
        assert_eq!(session.stats().warm_start_hits, 0);
        assert_eq!(
            session.stats().queries,
            session.stats().cache_hits
                + session.stats().warm_start_hits
                + session.stats().cache_misses,
            "queries must decompose into the three service paths"
        );
    }

    #[test]
    fn warm_start_works_with_the_verdict_cache_disabled() {
        let mut session = warm_session();
        session.set_caching_enabled(false);
        let _ = session.verify(&loose_spec()).expect("seed");
        let reused = session.verify(&wider_output_spec()).expect("reuse");
        assert!(reused.is_verified());
        assert_eq!(session.stats().warm_start_hits, 1);
        assert_eq!(session.stats().cache_misses, 1);
        assert_eq!(session.cache_len(), 0, "nothing cached when caching is off");
    }

    /// UNIT: the reuse predicate is the POSITIVE containment form, and refuses
    /// every shape the verifier would treat specially.
    #[test]
    fn discharge_predicate_refuses_empty_and_truncated_requirements() {
        let computed = vec![Bound::new(0.0, 1.0), Bound::new(-1.0, 1.0)];
        assert!(
            !computed_bounds_discharge(&computed, &[]),
            "an empty requirement is an InvalidSpec for the verifier, never a reuse"
        );
        assert!(
            !computed_bounds_discharge(
                &computed[..1],
                &[Bound::new(-1.0, 2.0), Bound::new(-2.0, 2.0)]
            ),
            "fewer computed bounds than requirements must never discharge"
        );
        // Exact equality is containment.
        assert!(computed_bounds_discharge(&computed, &computed));
        // One endpoint outside is enough to refuse.
        assert!(!computed_bounds_discharge(
            &computed,
            &[Bound::new(0.0, 1.0), Bound::new(-0.5, 1.0)]
        ));
        // Infinite requirements always contain a finite enclosure.
        assert!(computed_bounds_discharge(
            &computed,
            &[
                Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY),
                Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)
            ]
        ));
    }

    /// verify_many over a property sweep on ONE region: the first spec
    /// propagates, every later spec it discharges is reuse.
    #[test]
    fn verify_many_sweep_over_one_region_propagates_once() {
        let mut session = warm_session();
        let specs = vec![
            loose_spec(),        // required [-1, 2]
            wider_output_spec(), // required [-5, 5]
            VerificationSpec::new(vec![Bound::new(0.0, 1.0)], vec![Bound::new(-2.0, 3.0)])
                .expect("valid spec"),
        ];
        let results = session.verify_many(&specs);
        assert!(results
            .iter()
            .all(|r| r.as_ref().expect("no errors").is_verified()));
        assert_eq!(session.stats().queries, 3);
        assert_eq!(
            session.stats().cache_misses,
            1,
            "exactly one propagation for the whole sweep"
        );
        assert_eq!(session.stats().warm_start_hits, 2);
        assert_eq!(session.warm_start_len(), 1, "a single region");
    }
}
