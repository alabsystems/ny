// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dry-run refusal diagnostics: which nodes of THIS graph would a strict
//! [`translate`](super::translate) refuse, and why — all of them at once.
//!
//! ## The question neither existing predicate answers
//!
//! The bridge already ships two "can this be lowered?" surfaces, and each
//! answers a DIFFERENT question from the one a diagnostic caller has:
//!
//! - [`crate::coverage::is_translatable`] answers *"does a sound lowering exist
//!   IN PRINCIPLE"* — a property of the op taxonomy, deliberately independent of
//!   what [`super::translate_node`] implements today. Its own docs say so. It
//!   therefore returns `true` for `TraceOp::Topk { largest: false, sorted: true }`
//!   even though `translate` refuses that exact op by name, and `true` for a
//!   one-input `Gather` whose index tensor is missing. That is not a bug in
//!   `coverage`: a permissive caller needs the in-principle answer to decide
//!   whether an `OpaqueSkip` substitution is legitimate at all
//!   ([`crate::coverage::RefusalClass`] is derived from it).
//! - [`super::translate_with_metadata`] answers the right question, but only
//!   for the FIRST refused node, and only as stringly-typed
//!   [`NyError::UnsupportedOp`](ny_core::NyError::UnsupportedOp) text. A caller
//!   holding a graph with four unported ops must translate, read the message,
//!   patch or exclude that node, and translate again — four times.
//!
//! [`refusals`] is the missing third surface: the real translator's verdict,
//! for every node, in one pass, as data.
//!
//! ## Why it drives the real node loop
//!
//! The obvious implementation — a second `match` over [`TraceOp`] listing the
//! ops the bridge cannot lower — is exactly the failure mode this module
//! exists to remove. A downstream consumer (nn's `nn-verify`) hand-maintained
//! such a list and it drifted badly: it still named `Topk`, `Compare`,
//! `CompareTensor`, `Argmax`, `Argmin`, `ArgSort` and the whole scatter family
//! as unsupported long after this crate grew sound lowerings for them, so a
//! lenient caller was told a translatable graph was hopeless.
//!
//! A parallel classifier drifts because nothing forces it to agree. So
//! [`refusals`] does not classify anything. It runs
//! [`translate_impl`](super::translate_impl) — the ONE node loop, calling the
//! ONE per-node dispatch that the real translation calls — in a private
//! diagnostic mode, and reports what that dispatch actually returned. Porting
//! an op changes `refusals`' answer in the same commit, with no second list to
//! update.
//!
//! ## Why the dry run may ignore the soundness gate
//!
//! To find the refusal AFTER a refused node, the walk must get past that node,
//! which means standing something in its place. The diagnostic mode emits the
//! `LayerType::Unknown` / `OpaqueSkip` carrier UNCONDITIONALLY — ignoring both
//! the [`RefusalClass::Unported`](crate::coverage::RefusalClass) gate and the
//! [`super::UnportedOpPolicy`] policy gate that
//! [`super::degrade_to_opaque_skip`] enforces for real translations.
//!
//! That is sound ONLY because of containment, not because of the carrier: the
//! `GraphModel` the dry run assembles is unsound by construction and is dropped
//! unread inside [`collect`]. Nothing but the `Vec<NodeRefusal>` leaves this
//! module, and no public entry point can request the mode. `TranslateOptions`
//! gained no variant for it; the switch is a private parameter on a private
//! function.

use ny_core::NyError;

use super::{analyze_graph, translate_impl, validate_single_input_mode, TranslateOptions};
use crate::coverage::RefusalClass;
use crate::schema::ComputationGraph;

/// One node [`translate`](super::translate) would refuse, with the
/// classification a permissive caller needs to decide whether it may be
/// degraded.
///
/// The shape mirrors [`super::SkippedNode`] — the record a permissive
/// translation emits for a node it ACTUALLY degraded — minus `layer_name`
/// (a dry run emits no model, so there is no layer to name) and plus `class`
/// (a `SkippedNode` is `Unported` by construction; a `NodeRefusal` may be
/// either, and the difference is what a caller has to act on).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRefusal {
    /// `TraceNode::id` of the refused node.
    pub node_id: u64,
    /// `TraceNode::name` — the producer's own name for the node, i.e. the name
    /// to quote back to whoever traced the model.
    pub trace_name: String,
    /// Stable display name of the refused op — same source as
    /// [`SkippedNode::op`](super::SkippedNode::op), so the two records collate.
    pub op: &'static str,
    /// The `UnsupportedOp` message a strict translation would fail with,
    /// verbatim (`NyError::to_string()`, so it carries the
    /// `"Unsupported operation: "` prefix exactly as
    /// [`SkippedNode::reason`](super::SkippedNode::reason) does).
    pub reason: String,
    /// Whether a permissive translation may degrade this node.
    ///
    /// [`RefusalClass::Unported`] means
    /// [`TranslateOptions::permissive`](super::TranslateOptions::permissive)
    /// would replace the node with the vacuous `[-inf, +inf]` carrier and carry
    /// on. [`RefusalClass::Soundness`] means it would STILL hard-fail — so a
    /// caller that sees one of these knows that no policy makes this graph
    /// translate, and the trace itself has to change.
    pub class: RefusalClass,
}

/// Dry-run diagnostic: every node a strict
/// [`translate`](super::translate) would refuse, in graph order.
///
/// Single-input semantics, matching [`super::translate_with_metadata`] — the
/// multi-variable-input guard runs first. Use [`refusals_multi_input`] for a
/// graph with genuinely independent inputs.
///
/// # The guarantee
///
/// Write `R` for the returned vector.
///
/// - **Every entry is real.** Each one is a node the actual per-node dispatch
///   refused, carrying the actual message, because this walks the same node
///   loop `translate` walks. There is no parallel op classifier that could
///   disagree with the translator (which is the whole point — see the module
///   docs).
/// - **Reachable nodes only.** Dead nodes are skipped here exactly as
///   `translate` skips them, so an unported op in shadowed code is not
///   reported. Its refusal would not fail a real translation either.
/// - **Exhaustive, unless the walk had to stop.** `R` names every refused
///   reachable node UNLESS one of the two stopping conditions below fired, in
///   which case `R` is a prefix of the true answer, terminated at that node.
/// - **`R.is_empty()` does NOT mean the graph translates.** It means no
///   per-node refusal was found — see the caveats. Call
///   [`super::translate_with_metadata`] to learn whether the graph translates.
///
/// # Caveat 1 — a refusal the walk cannot step over
///
/// Standing the `OpaqueSkip` carrier in for a refused node is what lets the
/// walk continue, and the carrier has two structural preconditions — the node
/// needs at least one input (`OpaqueSkip` takes its output shape from its first
/// input edge), and every input edge must resolve to an already-translated
/// producer. A refused LEAF, or a refused node with a dangling edge, fails them.
///
/// Such a node **is recorded** and the walk then **stops**: `R` ends with it.
/// Everything downstream of it is unexamined, so a graph whose first refusal is
/// a leaf yields exactly one entry no matter how many later nodes would also
/// refuse.
///
/// # Caveat 2 — errors that are not refusals
///
/// [`NumericalInstability`](ny_core::NyError::NumericalInstability),
/// [`ModelLoad`](ny_core::NyError::ModelLoad) and
/// [`InternalError`](ny_core::NyError::InternalError) report a malformed,
/// numerically invalid or structurally broken graph — not a missing lowering.
/// They are **never** turned into a `NodeRefusal`, and they **stop** the walk:
/// continuing past a node whose topology or payload is already invalid would
/// produce refusals derived from garbage. `R` keeps whatever was collected
/// before that node.
///
/// # Caveat 3 — graph-level refusals raised before the node loop
///
/// These are properties of the whole graph, not of one node, so there is no
/// node to attribute them to and `R` comes back **empty**:
///
/// - more than one reachable variable input (the single-input aliasing guard).
///   Reach for [`refusals_multi_input`], which does not run that guard;
/// - the graph-wide F32 dtype gate (a reachable node whose declared output
///   dtype is not F32 and matches none of the four carve-outs);
/// - an empty graph, a duplicate node id, a failed topology validation, or a
///   marked output node that is not in the graph.
///
/// An empty `R` therefore has two readings — "nothing refuses" and "the graph
/// never reached the node loop" — and only `translate` tells them apart.
///
/// # Both refusal classes are reported
///
/// `R` includes [`RefusalClass::Soundness`] refusals as well as
/// [`RefusalClass::Unported`] ones, because a caller weighing a permissive
/// translation must know that a `Soundness` entry will hard-fail regardless of
/// policy. Filter on [`NodeRefusal::class`] to get the set
/// [`TranslateOptions::permissive`](super::TranslateOptions::permissive) would
/// actually degrade.
#[must_use]
pub fn refusals(graph: &ComputationGraph) -> Vec<NodeRefusal> {
    collect(graph, false)
}

/// Multi-input twin of [`refusals`], mirroring
/// [`super::translate_multi_input`].
///
/// Identical guarantee and identical caveats, with one difference: the
/// single-variable-input guard does not run, so a graph with several
/// independent inputs is walked instead of being refused wholesale before the
/// node loop (caveat 3's first bullet).
#[must_use]
pub fn refusals_multi_input(graph: &ComputationGraph) -> Vec<NodeRefusal> {
    collect(graph, true)
}

/// Shared body of [`refusals`] / [`refusals_multi_input`].
///
/// Reproduces the pre-loop sequence of the entry point it is standing in for
/// ([`super::translate_with_options`] resp.
/// [`super::translate_multi_input_with_options`]) so that a graph-level refusal
/// stops the dry run at exactly the point it would stop a real translation —
/// then hands the node loop the diagnostic sink.
fn collect(graph: &ComputationGraph, enable_multi_input: bool) -> Vec<NodeRefusal> {
    let mut found: Vec<NodeRefusal> = Vec::new();

    // Caveat 3: a graph-level refusal has no node to attribute, so the honest
    // answer is the empty list rather than an invented entry.
    let Ok(analysis) = analyze_graph(graph) else {
        return found;
    };
    if !enable_multi_input && validate_single_input_mode(&analysis).is_err() {
        return found;
    }

    // `options` is inert on this path: the diagnostic branch of the node loop
    // short-circuits before `degrade_to_opaque_skip` ever consults a policy.
    // Passing `strict()` keeps that obvious rather than implying the dry run
    // is "a permissive translation with extra reporting" — it is not; it
    // degrades nodes a permissive translation would refuse outright.
    //
    // THE MODEL BUILT HERE IS DELIBERATELY UNSOUND (soundness-class refusals
    // were replaced with vacuous carriers to keep the walk going), which is
    // why it is dropped unread and why no signature in this module can return
    // it. Only `found` escapes. See `translate_impl`'s `diagnostics` parameter.
    drop(translate_impl(
        graph,
        enable_multi_input,
        &analysis,
        TranslateOptions::strict(),
        Some(&mut found),
    ));

    found
}

/// Is `error` a REFUSAL — a node the bridge declined to lower — as opposed to a
/// report that the graph itself is invalid?
///
/// The single source of the rule, used by the node loop's diagnostic branch
/// (which records only refusals) and by
/// [`degrade_to_opaque_skip`](super::degrade_to_opaque_skip) (which degrades
/// only refusals). Only [`NyError::UnsupportedOp`] qualifies:
/// `NumericalInstability` (an epsilon gate, a non-finite weight), `ModelLoad`
/// (a malformed or oversized payload) and `InternalError` (a broken topology, a
/// trace that contradicts itself) all say the graph is wrong, not that a port is
/// missing — and neither vacuous-ing nor tabulating one of those would be
/// honest.
pub(super) const fn is_refusal(error: &NyError) -> bool {
    matches!(error, NyError::UnsupportedOp(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::is_translatable;
    use crate::schema::{DType, NodeId, TraceNode, TraceOp};
    use crate::translate::{translate, translate_multi_input, translate_with_metadata};

    fn node(id: u64, name: &str, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
        TraceNode::new(
            NodeId(id),
            name,
            op,
            inputs.iter().map(|&i| NodeId(i)).collect(),
            shape.to_vec(),
            DType::F32,
        )
    }

    /// `Topk` with `largest = false`: a sound lowering exists in principle
    /// (`coverage` says `Exact`), but `TopkLayer` models only the sorted-
    /// descending k-largest selection, so `translate` refuses it by name. The
    /// canonical witness for "the two predicates answer different questions".
    fn k_smallest_topk() -> TraceOp {
        TraceOp::Topk {
            k: 2,
            dim: 0,
            largest: false,
            sorted: true,
        }
    }

    /// A one-input `Gather` — the index tensor is missing, so the lowering is
    /// impossible for a reason that has nothing to do with `Gather`'s
    /// taxonomy entry (also `Exact`). A second, structurally different
    /// refusal for the multi-refusal walk.
    fn indexless_gather() -> TraceOp {
        TraceOp::Gather { dim: 0 }
    }

    /// `MoeGating`: refused for SOUNDNESS (data-dependent expert routing), so
    /// no policy degrades it. Present to prove `class` discriminates.
    fn moe_gating() -> TraceOp {
        TraceOp::MoeGating {
            num_experts: 2,
            top_k: 1,
        }
    }

    /// `Input -> Relu -> Sigmoid`: nothing in it refuses.
    fn clean_graph() -> ComputationGraph {
        ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "act", TraceOp::Relu, &[0], &[4]),
            node(2, "sig", TraceOp::Sigmoid, &[1], &[4]),
        ])
    }

    /// `Input -> Relu -> <op> -> Relu`, supported except for the one node under
    /// test.
    fn graph_with(op: TraceOp, out_shape: &[usize]) -> ComputationGraph {
        ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "pre", TraceOp::Relu, &[0], &[4]),
            node(2, "subject", op, &[1], out_shape),
            node(3, "post", TraceOp::Relu, &[2], out_shape),
        ])
    }

    /// (1) A graph that translates cleanly has NO refusals.
    #[test]
    fn a_clean_graph_has_no_refusals() {
        let graph = clean_graph();
        translate(&graph).expect("the fixture must translate, or the test proves nothing");
        assert!(
            refusals(&graph).is_empty(),
            "a translatable graph must report no refusals: {:?}",
            refusals(&graph)
        );
        assert_eq!((refusals_multi_input(&graph)).len(), 0);
    }

    /// (2) One refused node yields exactly one refusal, fully populated.
    #[test]
    fn one_refused_node_yields_one_fully_populated_refusal() {
        let graph = graph_with(k_smallest_topk(), &[2]);
        let found = refusals(&graph);
        assert_eq!(found.len(), 1, "one refused node, one refusal: {found:?}");

        let refusal = &found[0];
        assert_eq!(refusal.node_id, 2);
        assert_eq!(refusal.trace_name, "subject");
        assert_eq!(refusal.op, "Topk");
        assert_eq!(
            refusal.class,
            RefusalClass::Unported,
            "a sound lowering exists in principle, so this is a missing layer"
        );
        assert!(
            refusal.reason.contains("largest=false") && refusal.reason.contains("k smallest"),
            "the reason is the translator's own message, not a generic label: {}",
            refusal.reason
        );
    }

    /// (3) THE ANTI-DRIFT TEST. A graph with several distinct refused nodes
    /// reports EVERY one of them, where `translate` reports only the first.
    /// This is the property the hand-maintained downstream list could never
    /// deliver and the reason the API exists.
    #[test]
    fn every_refused_node_is_reported_not_just_the_first() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "topk", k_smallest_topk(), &[0], &[2]),
            node(2, "act", TraceOp::Relu, &[1], &[2]),
            node(3, "gather", indexless_gather(), &[2], &[2]),
            node(4, "moe", moe_gating(), &[3], &[2]),
            node(5, "post", TraceOp::Relu, &[4], &[2]),
        ]);

        // `translate` sees the first refusal and stops. That is the behaviour
        // being improved on, so pin it.
        let first_error = translate(&graph).expect_err("the graph does not translate");
        assert!(
            matches!(&first_error, NyError::UnsupportedOp(m) if m.contains("Topk")),
            "translate reports only the FIRST refusal: {first_error:?}"
        );

        let found = refusals(&graph);
        let named: Vec<&str> = found.iter().map(|r| r.op).collect();
        assert_eq!(
            named,
            vec!["Topk", "Gather", "MoeGating"],
            "all three refusals, in graph order: {found:?}"
        );
        assert_eq!(
            found.iter().map(|r| r.node_id).collect::<Vec<_>>(),
            vec![1, 3, 4]
        );

        // And the class discriminates: the first two are missing lowerings a
        // permissive translation would degrade, the third is a soundness
        // refusal that hard-fails under every policy.
        assert_eq!(found[0].class, RefusalClass::Unported);
        assert_eq!(found[1].class, RefusalClass::Unported);
        assert_eq!(
            found[2].class,
            RefusalClass::Soundness,
            "MoeGating's routing is data-dependent; no policy degrades it"
        );
        assert_eq!(
            found.iter().filter(|r| r.class.is_degradable()).count(),
            2,
            "exactly the two nodes permissive mode could stand in for"
        );

        // The reachable-node rule still holds: the intervening Relu nodes are
        // fine and contribute nothing.
        assert!(found.iter().all(|r| r.op != "Relu"));
    }

    /// (4) THE PIN. For a single-refusal graph the recorded reason is the
    /// strict translation's error text, verbatim. If `refusals` ever grew its
    /// own message table this fails immediately.
    #[test]
    fn the_reason_is_the_strict_translations_own_error() {
        for (label, op, shape) in [
            ("Topk", k_smallest_topk(), vec![2usize]),
            ("Gather", indexless_gather(), vec![4usize]),
            ("MoeGating", moe_gating(), vec![4usize]),
        ] {
            let graph = graph_with(op, &shape);
            let error =
                translate_with_metadata(&graph).expect_err("the subject op must be refused");
            let found = refusals(&graph);
            assert_eq!(found.len(), 1, "{label}: {found:?}");
            assert_eq!(
                found[0].reason,
                error.to_string(),
                "{label}: the dry run must quote the real translator"
            );
            assert!(
                matches!(&error, NyError::UnsupportedOp(msg) if found[0].reason.contains(msg)),
                "{label}: and it must be an UnsupportedOp: {error:?}"
            );
        }
    }

    /// (5) The documented reason BOTH predicates exist: `coverage` answers
    /// "could this be lowered soundly at all", `refusals` answers "does the
    /// translator lower it today". They are supposed to disagree here, and a
    /// caller that substitutes one for the other gets the wrong answer.
    #[test]
    fn refusals_disagrees_with_is_translatable_exactly_where_it_should() {
        // Disagreement: a sound lowering exists in principle, but not today.
        for op in [k_smallest_topk(), indexless_gather()] {
            assert!(
                is_translatable(&op),
                "coverage: a sound lowering exists in principle for {op:?}"
            );
            let graph = graph_with(op, &[2]);
            let found = refusals(&graph);
            assert_eq!(
                found.len(),
                1,
                "but translate refuses it today, so refusals reports it: {found:?}"
            );
            assert_eq!(found[0].class, RefusalClass::Unported);
        }

        // Agreement: an op that IS lowered today is refused by neither.
        let sorted_descending_topk = TraceOp::Topk {
            k: 2,
            dim: 0,
            largest: true,
            sorted: true,
        };
        assert!(is_translatable(&sorted_descending_topk));
        assert!(
            refusals(&graph_with(sorted_descending_topk, &[2])).is_empty(),
            "the flag combination TopkLayer models lowers, so nothing is reported"
        );

        // Agreement the other way: `coverage` says never-lowerable, and the
        // translator agrees — but only `refusals` says WHICH node.
        assert!(!is_translatable(&moe_gating()));
        let found = refusals(&graph_with(moe_gating(), &[4]));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].class, RefusalClass::Soundness);
    }

    /// Caveat 1: a refused node whose `OpaqueSkip` carrier cannot be emitted —
    /// here a refused LEAF — is recorded, and then the walk stops. The list is
    /// a prefix, and the doc says so rather than claiming exhaustiveness.
    #[test]
    fn a_refused_leaf_is_recorded_and_then_stops_the_walk() {
        let graph = ComputationGraph::from_nodes(vec![
            // Zero inputs: `OpaqueSkip` takes its output shape from its first
            // input edge, so there is nothing to stand in for this node.
            node(0, "leaf", k_smallest_topk(), &[], &[2]),
            node(1, "act", TraceOp::Relu, &[0], &[2]),
            // Would refuse too — but is never reached.
            node(2, "gather", indexless_gather(), &[1], &[2]),
        ]);
        let found = refusals(&graph);
        assert_eq!(
            found.len(),
            1,
            "the leaf refusal is recorded, and nothing after it is: {found:?}"
        );
        assert_eq!(found[0].op, "Topk");
        assert_eq!(found[0].node_id, 0);
    }

    /// Caveat 2: a non-`UnsupportedOp` failure is not a refusal. It is not
    /// recorded, and it stops the walk — earlier refusals survive.
    #[test]
    fn a_non_refusal_error_is_not_recorded_and_stops_the_walk() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "topk", k_smallest_topk(), &[0], &[2]),
            // `InternalError`: the recorded output extent contradicts the op's
            // own k, which is a malformed trace rather than a missing port.
            node(
                2,
                "contradictory",
                TraceOp::Topk {
                    k: 3,
                    dim: 0,
                    largest: true,
                    sorted: true,
                },
                &[1],
                &[7],
            ),
            node(3, "gather", indexless_gather(), &[2], &[7]),
        ]);
        assert!(
            matches!(
                translate(&graph).expect_err("malformed"),
                NyError::UnsupportedOp(_)
            ),
            "translate still trips on the earlier Topk refusal"
        );

        let found = refusals(&graph);
        assert_eq!(
            found.len(),
            1,
            "the genuine refusal survives; the malformed node is neither \
             recorded nor walked past: {found:?}"
        );
        assert_eq!(found[0].node_id, 1);
        assert!(
            found.iter().all(|r| r.node_id != 2),
            "an InternalError must never be laundered into a NodeRefusal"
        );
    }

    /// Caveat 3: a graph-level refusal raised BEFORE the node loop has no node
    /// to attribute, so `refusals` reports nothing — and the multi-input twin,
    /// which does not run that guard, finds the per-node refusal underneath it.
    #[test]
    fn the_variable_input_guard_yields_no_refusals_but_the_multi_input_twin_walks() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "a", TraceOp::Input, &[], &[4]),
            node(1, "b", TraceOp::Input, &[], &[4]),
            node(2, "sum", TraceOp::Add, &[0, 1], &[4]),
            node(3, "topk", k_smallest_topk(), &[2], &[2]),
        ]);

        assert!(
            matches!(
                translate(&graph).expect_err("two variable inputs"),
                NyError::UnsupportedConfiguration(_)
            ),
            "the guard, not a node, is what refuses the single-input path"
        );
        assert!(
            refusals(&graph).is_empty(),
            "an empty list here means 'no PER-NODE refusal was found', never \
             'this graph translates'"
        );

        let found = refusals_multi_input(&graph);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].op, "Topk");
        assert_eq!(found[0].node_id, 3);
        assert!(
            translate_multi_input(&graph).is_err(),
            "and the multi-input translation really does refuse that node"
        );
    }

    /// Caveat 3, the other pre-loop gates: an empty graph, a broken topology
    /// and the graph-wide F32 dtype gate all return the empty list.
    #[test]
    fn pre_loop_graph_gates_all_return_the_empty_list() {
        assert_eq!((refusals(&ComputationGraph::from_nodes(vec![]))).len(), 0);

        // Duplicate node id: `analyze_graph` refuses before any node is walked.
        let duplicate = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(0, "x_again", TraceOp::Relu, &[], &[4]),
        ]);
        assert_eq!((refusals(&duplicate)).len(), 0);
        assert_eq!((refusals_multi_input(&duplicate)).len(), 0);

        // Graph-wide dtype gate: a reachable non-F32 node that matches none of
        // the carve-outs. The refusal IS an `UnsupportedOp`, but it is raised
        // over the node list rather than by the dispatch, so no node owns it.
        let mut non_f32 = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "act", TraceOp::Relu, &[0], &[4]),
            node(2, "topk", k_smallest_topk(), &[1], &[2]),
        ]);
        non_f32.nodes[1].output_dtype = DType::F16;
        assert!(
            matches!(
                translate(&non_f32).expect_err("non-F32 fails closed"),
                NyError::UnsupportedOp(m) if m.contains("output dtype")
            ),
            "the dtype gate is the refusal, not the Topk below it"
        );
        assert!(
            refusals(&non_f32).is_empty(),
            "a graph-wide gate has no node to attribute"
        );
    }

    /// Dead nodes are invisible to `refusals`, exactly as they are to
    /// `translate` — an unported op in code shadowed by a composite must not be
    /// reported as a blocker for a graph that translates fine.
    #[test]
    fn unreachable_refused_nodes_are_not_reported() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            // Dead: consumes x, feeds nothing on the output path.
            node(1, "dead_topk", k_smallest_topk(), &[0], &[2]),
            node(2, "act", TraceOp::Relu, &[0], &[4]),
        ]);
        translate(&graph).expect("a dead refused node does not fail translation");
        assert!(
            refusals(&graph).is_empty(),
            "and it is not reported as a blocker either"
        );
    }

    /// The dry run must not perturb the real translator: running `refusals`
    /// before and after a translation changes nothing, and the translation's
    /// own verdict is unchanged. (`refusals` takes `&ComputationGraph` and
    /// builds its own throwaway state, so this is a guard against a future
    /// refactor sharing mutable context between the two.)
    #[test]
    fn the_dry_run_leaves_the_real_translation_untouched() {
        let graph = clean_graph();
        let before = translate(&graph).expect("translates").network.layers.len();
        let _ = refusals(&graph);
        let after = translate(&graph)
            .expect("still translates")
            .network
            .layers
            .len();
        assert_eq!(before, after);

        let refused = graph_with(k_smallest_topk(), &[2]);
        assert!(translate(&refused).is_err());
        let _ = refusals(&refused);
        assert!(
            translate(&refused).is_err(),
            "the dry run must not make a refused graph translate"
        );
    }

    /// `is_refusal` documents the discriminator the node loop applies; keep it
    /// honest so the doc cannot rot into a lie.
    #[test]
    fn only_unsupported_op_counts_as_a_refusal() {
        assert!(is_refusal(&NyError::UnsupportedOp("x".to_string())));
        assert!(!is_refusal(&NyError::InternalError("x".to_string())));
        assert!(!is_refusal(&NyError::UnsupportedConfiguration(
            "x".to_string()
        )));
    }
}
