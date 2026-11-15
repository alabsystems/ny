// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #admission-precision — the lanes' own ROUTE predicates, as one callable
//! implementation the lanes and the router both call. PURE: no clock, no
//! environment, no filesystem, no global.
//!
//! # Why these live here
//!
//! A routable capability's consumer sits deep inside a parent lane, and the
//! parent reaches it only on some rows: the disjunctive lane only on
//! multi-clause or boxed-clause properties, a non-conjunctive graph
//! multi-objective BaB only on a top-level OR of two or more atoms, the graph
//! input-split lane only on a model the loader lowers to a graph (or a
//! property the CLI converts a sequential model for), the lA warm-start
//! frontier only on a graph with an n-ary node. Those checks used to be
//! inline expressions in the lane bodies, so the router could not ask them at
//! start and admitted every row its operator precondition allowed
//! (#admission-census: 4599 moat rows for three of the four capabilities
//! below).
//!
//! Each check is now a named function here. The lane's inline check CALLS it
//! (`ny-cli`: `dispatch.rs` `routes_to_relational_verifier`,
//! `verify/mod.rs` `verify_relational_constraints_with_ledger` and
//! `should_upgrade_sequential_conjunction_to_graph`, `verify/graph.rs`
//! `verify_graph_relational`, `model_load.rs` `load_model`; source ratchets in
//! `ny-cli`'s `router::admission_tests` pin every call), with the
//! specification shape the lanes read through the SAME extractor the router
//! uses. The router composes them into one [`AdmitSite`] per capability
//! (`AdmitWhen::site`), evaluated on the row's facts at start.
//!
//! # Necessary conditions, policy omitted
//!
//! An [`AdmitSite`] DECLINES a row only when the consumer cannot engage on it
//! under ANY configuration: the policy-dependent checks the lanes also make
//! (the branching method, `reorder_bab`, `optimize_disjuncts_separately`,
//! `gpu_bab`, a yaml rewrite such as the VGG MaxPool treatment or the softmax
//! decomposition) are omitted, exactly as the census omits them from the
//! margin-row probe. The operator kinds are the scan's conservative
//! signature (a superset of what the loader builds), so a PRESENCE test over
//! them never refuses a row whose graph has the kind; no test here refuses a
//! row because a kind is present. An unknown shape (the property was not
//! parsed at start) defers the specification part to the consumer's call
//! site, which re-checks it with the same function.
//!
//! A wrong decline can only cost reach: the router then does not route the
//! capability, so the consumer never engages through a route on that row,
//! and the census (the router's own decision) leaves the row out of the seal's
//! moat in agreement. Nothing here names or moves a verdict.

use serde::{Deserialize, Serialize};

use super::decl::{AdmitSite, OpKind, OpSet};
use super::spec::SpecShape;

// ---- the lanes' own checks ---------------------------------------------------

/// `dispatch.rs` `routes_to_relational_verifier`: the shared BaB dispatch
/// selects the relational verifier (`verify_relational_constraints_with_ledger`)
/// rather than `verify_standard` — more than one output atom, an atom that
/// compares two outputs, or a boxed-clause disjunction.
#[must_use]
pub const fn relational_verifier(spec: &SpecShape) -> bool {
    spec.atoms > 1 || spec.relational_atoms || spec.boxed_clause_disjunction
}

/// `verify/mod.rs` `verify_relational_constraints_with_ledger`: the
/// relational verifier forwards the row to the multi-clause disjunctive lane
/// (`verify_multi_clause_disjunction`) — a top-level OR of two or more clauses,
/// or clauses with their own input boxes (even one).
#[must_use]
pub const fn disjunctive_lane(spec: &SpecShape) -> bool {
    spec.multi_clause_disjunction || spec.boxed_clause_disjunction
}

/// The row reaches `verify_multi_clause_disjunction`: the dispatch selects the
/// relational verifier AND it forwards to the disjunctive lane.
#[must_use]
pub const fn reaches_disjunctive_lane(spec: &SpecShape) -> bool {
    relational_verifier(spec) && disjunctive_lane(spec)
}

/// How a graph multi-objective BaB aggregates its objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoAggregation {
    /// Any objective proved impossible closes a domain (the property is an AND).
    Conjunctive,
    /// Every objective must be proved impossible (the property is an OR).
    Disjunctive,
}

/// `verify/graph.rs` `verify_graph_relational`: the multi-objective graph BaB
/// runs over two or more atoms, conjunctive unless the clauses are a top-level
/// OR (`None`: one atom, the per-constraint path). The lane takes the
/// disjunctive aggregation on the ReLU-split route only.
#[must_use]
pub const fn graph_multi_objective(spec: &SpecShape) -> Option<MoAggregation> {
    if spec.atoms < 2 {
        None
    } else if spec.disjunction {
        Some(MoAggregation::Disjunctive)
    } else {
        Some(MoAggregation::Conjunctive)
    }
}

/// `verify/mod.rs` `should_upgrade_sequential_conjunction_to_graph`: a
/// sequential model's conjunction of two or more atoms is upgraded to the
/// graph multi-objective lane — always on the ReLU-split route, and on the
/// input-split route unless the sequential same-LHS reduction applies.
#[must_use]
pub const fn sequential_conjunction_upgrade(spec: &SpecShape, use_relu_split: bool) -> bool {
    !spec.disjunction && spec.atoms >= 2 && (use_relu_split || !spec.same_lhs_reducible)
}

/// `model_load.rs`: a per-clause-box disjunction with at least this many
/// boxed clauses routes a sequential-capable net to the graph for the batched
/// box-refinement screen.
pub const HUGE_BOXED_CLAUSE_DISJUNCTION_MIN: u64 = 512;

/// `model_load.rs`: whether `boxed_clauses` boxed clauses make a massive
/// per-clause-box disjunction.
#[must_use]
pub const fn huge_boxed_clause_disjunction(boxed_clauses: u64) -> bool {
    boxed_clauses >= HUGE_BOXED_CLAUSE_DISJUNCTION_MIN
}

/// The three terms of the loader's graph decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphRouteTerms {
    /// The graph has a node with two or more activation inputs.
    pub n_ary: bool,
    /// A convolution routes the model to the graph.
    pub conv: bool,
    /// A massive per-clause-box disjunction routes it there.
    pub huge_boxed_disjunction: bool,
}

/// `model_load.rs` `load_model`: the loader builds a `GraphNetwork` (not the
/// sequential lowering). MONOTONE in every term, so the router, which feeds
/// it an upper bound of each term the lane computes, answers `true` whenever
/// the lane does.
#[must_use]
pub const fn graph_network(terms: GraphRouteTerms) -> bool {
    terms.n_ary || terms.conv || terms.huge_boxed_disjunction
}

// ---- the router's bounds of the lanes' model terms ---------------------------

/// Operator kinds that can put a node with two or more activation inputs into
/// the graph the loader decides on: a computed residual join, a concat, a
/// bilinear product, a normalization (decomposed into primitive binary
/// nodes), an unrecognised operator, and a pool the VGG MaxPool treatment (a
/// yaml rewrite that runs BEFORE the graph decision) rewrites into primitives.
/// Policy is omitted, so the pool counts.
pub const N_ARY_AT_LOAD: OpSet = OpSet::EMPTY
    .with(OpKind::Residual)
    .with(OpKind::Concat)
    .with(OpKind::Bilinear)
    .with(OpKind::Norm)
    .with(OpKind::Pool)
    .with(OpKind::Other);

/// Operator kinds a rewrite AFTER the graph decision can turn into binary
/// nodes on a graph model (the softmax decomposition, a yaml rewrite).
pub const N_ARY_AFTER_LOAD: OpSet = OpSet::EMPTY.with(OpKind::Softmax);

/// #kfsb-production: operator kinds whose presence lets the graph carry a
/// ReLU node a ReLU-split selector can split: a ReLU, a pool the VGG MaxPool
/// treatment (a yaml rewrite) turns into primitives, and an unrecognised
/// operator the loader may lower to one. Policy is omitted, so the pool and
/// the unknown kind count (an over-wide answer only widens the moat).
pub const RELU_SPLIT_KINDS: OpSet = OpSet::EMPTY
    .with(OpKind::Relu)
    .with(OpKind::Pool)
    .with(OpKind::Other);

/// Operator kinds whose presence lets the loader route a model to the graph
/// for its convolutions.
pub const CONV_ROUTE: OpSet = OpSet::EMPTY.with(OpKind::Conv).with(OpKind::ConvTranspose);

const fn any_of(ops: OpSet, kinds: OpSet) -> bool {
    ops.0 & kinds.0 != 0
}

/// The router's upper bound of the loader's graph decision
/// ([`graph_network`]) on the scan's operator signature and the row's shape
/// (`None`: the shape is unknown and no operator term decides it).
#[must_use]
pub const fn graph_network_possible(ops: OpSet, spec: Option<&SpecShape>) -> Option<bool> {
    let n_ary = any_of(ops, N_ARY_AT_LOAD);
    let conv = any_of(ops, CONV_ROUTE);
    if n_ary || conv {
        return Some(true);
    }
    match spec {
        Some(spec) => Some(graph_network(GraphRouteTerms {
            n_ary,
            conv,
            huge_boxed_disjunction: huge_boxed_clause_disjunction(spec.boxed_clauses),
        })),
        None => None,
    }
}

/// Whether a graph BaB can hold an n-ary node: one from the load, or one a
/// post-load rewrite makes on a model the loader lowered to a graph.
#[must_use]
pub const fn dag_possible(ops: OpSet, spec: Option<&SpecShape>) -> Option<bool> {
    if any_of(ops, N_ARY_AT_LOAD) {
        Some(true)
    } else if any_of(ops, N_ARY_AFTER_LOAD) {
        graph_network_possible(ops, spec)
    } else {
        Some(false)
    }
}

// ---- the site of each routable consumer ----------------------------------------

const fn and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

const fn or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// `pred(spec)` when the shape is known, else undecided.
fn on_spec(spec: Option<&SpecShape>, pred: fn(&SpecShape) -> bool) -> Option<bool> {
    spec.map(pred)
}

const fn disjunctive_mo_on_graph_spec(spec: &SpecShape) -> bool {
    relational_verifier(spec)
        && matches!(
            graph_multi_objective(spec),
            Some(MoAggregation::Disjunctive)
        )
}

const fn sequential_input_split_upgrade(spec: &SpecShape) -> bool {
    relational_verifier(spec) && sequential_conjunction_upgrade(spec, false)
}

/// #kfsb-production: a sequential model's conjunction upgraded to the graph
/// multi-objective lane on the ReLU-split route (the relational verifier
/// sends it there; the upgrade always applies on that route).
const fn sequential_relu_split_upgrade(spec: &SpecShape) -> bool {
    relational_verifier(spec) && sequential_conjunction_upgrade(spec, true)
}

/// Whether the consumer at `site` can be reached on a row with operator
/// signature `ops` and specification shape `spec` (`None`: the shape is
/// unknown at start and the operators do not decide it — deferred to the
/// consumer's own call site). Kleene logic over the lane predicates above.
#[must_use]
pub fn site_reachable(site: AdmitSite, ops: OpSet, spec: Option<&SpecShape>) -> Option<bool> {
    let graph = graph_network_possible(ops, spec);
    match site {
        // `verify_multi_clause_disjunction`'s forward root screen runs on a
        // `GraphNetwork` only (no conversion precedes it).
        AdmitSite::DisjunctiveLaneGraph => and(on_spec(spec, reaches_disjunctive_lane), graph),
        // A non-conjunctive graph multi-objective BaB: `verify_graph_relational`
        // over a top-level OR of >= 2 atoms on a graph model, or the
        // per-disjunct multi-objective inside the disjunctive lane (which
        // converts a sequential model itself).
        AdmitSite::DisjunctiveMultiObjective => or(
            on_spec(spec, reaches_disjunctive_lane),
            and(on_spec(spec, disjunctive_mo_on_graph_spec), graph),
        ),
        // The graph input-split lane: a graph model (any property), or a
        // sequential model the CLI converts — the input-split conjunction
        // upgrade, or the disjunctive lane's unified input-split BaB.
        AdmitSite::GraphInputSplit => or(
            graph,
            or(
                on_spec(spec, sequential_input_split_upgrade),
                on_spec(spec, reaches_disjunctive_lane),
            ),
        ),
        // The #warm-dag guard refuses a seed only above an n-ary node; a
        // converted sequential model is a chain.
        AdmitSite::DagGraph => dag_possible(ops, spec),
        // #kfsb-production: the shared graph kFSB selector runs inside a
        // graph ReLU-split BaB — the DomainList lane on a graph model (any
        // property), or the conjunctive multi-objective lane on a sequential
        // model the CLI converts (the ReLU-split conjunction upgrade, or the
        // disjunctive lane's per-disjunct graph BaB).
        // A ReLU-split selector needs a ReLU to split (a kind in
        // `RELU_SPLIT_KINDS`); that is an operator PRESENCE test.
        AdmitSite::GraphReluSplit => and(
            Some(any_of(ops, RELU_SPLIT_KINDS)),
            or(
                graph,
                or(
                    on_spec(spec, sequential_relu_split_upgrade),
                    on_spec(spec, reaches_disjunctive_lane),
                ),
            ),
        ),
    }
}
