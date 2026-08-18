// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph-MIP LEAF oracle seam (increment 6, `docs/GRAPH_MIP_LEAF_SOLVER.md`).
//!
//! The per-subdomain exact-MIP escalation: when the graph ReLU-split BaB is
//! about to REQUEUE an undecided child, it may consult an externally attached
//! oracle that encodes the child's subdomain (split premises PINNED, sound
//! per-node boxes as big-M ranges) as an exact MIP and solves the undecided
//! spec rows. The encoder + solver live in ny-cli/ny-mip, which ny-propagate
//! cannot depend on — hence this trait seam, mirroring the
//! [`JointMarginCloser`](super::engine::JointMarginCloser) precedent (a
//! runtime-attached `Arc` closer on `BetaCrownVerifier`, preserved by
//! `with_config_from`, `None` = byte-identical default).
//!
//! # Soundness contract (the implementor's obligations)
//!
//! - `VerifiedAllRows` from [`GraphMipLeafOracle::solve_leaf`] may be returned
//!   ONLY when every row in [`GraphMipLeafRequest::rows`] is proven
//!   infeasible-to-violate on the subdomain with CERTIFIED evidence (verified
//!   Farkas certificate — the 0-wrong moat).
//! - `VerifiedAllRows` from [`GraphMipLeafOracle::solve_input_leaf`] may be
//!   returned ONLY when every row in [`GraphInputLeafRequest::objectives`] is
//!   strictly certified over the complete input box by that implementation's
//!   independently audited proof system. The loop then counts the child
//!   verified instead of queueing it. The packed grouped layout does not weaken
//!   this foundation contract to "one row per clause": doing that safely will
//!   require a future typed certified-row mask whose coverage the caller can
//!   validate before granting authority.
//! - `Violated` may be returned ONLY for a witness that was clamped into the
//!   domain's input box and CONFIRMED by an independent forward pass through
//!   the original graph. It is ADVISORY: the loop logs it and REQUEUES the
//!   child (never drops it, never `Verified`) — sat-side reporting stays with
//!   the attack lanes. A well-behaved oracle latches itself off afterwards.
//! - Anything else — timeout, uncertified UNSAT, unconfirmed witness, any
//!   internal failure — MUST be `Undecided`: the child is queued exactly as
//!   without the oracle. The oracle is therefore strictly additive: it can
//!   convert "requeue" into "verified", never flip or discard a BaB decision.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ny_tensor::BoundedTensor;

use crate::{GraphNetwork, PhaseBudgetConfig};

/// One BaB split premise, in the external tooling's vocabulary
/// (`Relu_31:1337:I` / emit_hard_six `prem=[j±@0]`): the ReLU node whose
/// pre-activation neuron `neuron_idx` is constrained to the active
/// (`x >= 0`) or inactive (`x <= 0`) piece on this subdomain.
#[derive(Debug, Clone)]
pub struct LeafSplit {
    /// The ReLU node's graph name (the premise clamps its INPUT node's box).
    pub relu_node: String,
    /// Flat neuron index within the ReLU's pre-activation tensor.
    pub neuron_idx: usize,
    /// `true` = active piece (`x >= 0`), `false` = inactive (`x <= 0`).
    pub is_active: bool,
}

/// A lightweight, input-box-only leaf request.
///
/// Unlike [`GraphMipLeafRequest`], this request deliberately carries no
/// intermediate node bounds or split premises.  It is offered before the
/// Graph-MIP leaf path collects its big-M ranges, allowing an independently
/// attached certified verifier to replay the complete graph directly on the
/// exact input sub-box without paying for an unrelated bound collection.
///
/// The objective rows use the grouped disjunctive layout described by
/// `clause_sizes`.  Returning [`GraphMipLeafVerdict::VerifiedAllRows`] from
/// [`GraphMipLeafOracle::solve_input_leaf`] requires a certified strict proof
/// of *every* objective row in this request over the complete input box.  Any
/// unsupported layout, timeout, internal error, or partial proof must return
/// [`GraphMipLeafVerdict::Undecided`]. In particular, a proof of only one row
/// per clause has no authority at this seam yet: a future extension must carry
/// a typed certified-row mask and validate its grouped coverage explicitly.
///
/// [`GraphInputLeafRequest::advisory_objective_bounds`] is scheduling metadata
/// only. It lets an implementation decline leaves far from its useful frontier
/// before launching an expensive independent proof. Those carried BaB bounds
/// have no certificate identity at this trait seam and grant no verdict
/// authority: an implementation that elects to run must still certify every
/// objective row over [`GraphInputLeafRequest::input_bounds`] independently.
pub struct GraphInputLeafRequest<'a> {
    /// The ORIGINAL graph whose reachable outputs must be enclosed.
    pub graph: &'a GraphNetwork,
    /// The subdomain's exact input box.
    pub input_bounds: &'a BoundedTensor,
    /// Sign-normalized objective rows over the graph output.
    pub objectives: &'a Array2<f32>,
    /// Advisory lower/upper bounds for `objectives`, in the same row order.
    ///
    /// These are borrowed from the BaB domain solely for scheduling (for
    /// example, bounded near-frontier admission). They may be loose and are
    /// not authenticated proof artifacts at this seam. They MUST NOT justify
    /// `VerifiedAllRows`, replace an implementation's independent row replay,
    /// or weaken the all-rows contract. Implementations must validate shape
    /// and finiteness before using them even for scheduling.
    pub advisory_objective_bounds: &'a [(f32, f32)],
    /// One strict lower-bound threshold per objective row.
    pub thresholds: &'a [f32],
    /// Clause widths for the packed grouped-disjunctive objective layout.
    pub clause_sizes: &'a [usize],
    /// The subdomain's BaB depth.
    pub depth: usize,
    /// Wall-clock deadline for the whole verification.  A proof completed at
    /// or after this instant has no verdict authority.
    pub deadline: Option<Instant>,
}

/// Everything the leaf oracle needs to encode + solve one subdomain.
pub struct GraphMipLeafRequest<'a> {
    /// The ORIGINAL graph (also the revalidation forward for any witness).
    pub graph: &'a GraphNetwork,
    /// The subdomain's input box (exact; never inflated by the encoder).
    pub input_bounds: &'a BoundedTensor,
    /// The subdomain's sound per-node bounds (PRE-activation boxes keyed by
    /// node name). NOTE: split premises are NOT clamped into these (the BaB
    /// enforces them at relaxation time); the oracle must apply
    /// [`GraphMipLeafRequest::splits`] itself (the emit_hard_six `clamp()`
    /// step) before encoding.
    pub node_bounds: &'a HashMap<String, Arc<BoundedTensor>>,
    /// The subdomain's split premises (from `GraphSplitHistory::constraints`).
    pub splits: Vec<LeafSplit>,
    /// The UNDECIDED spec rows: `(objective coefficients over the graph
    /// output, threshold)`. A row is verified on the subdomain iff
    /// `objective · y > threshold` for every reachable output `y` — i.e. iff
    /// the decision MIP `objective · y <= threshold` is infeasible.
    pub rows: Vec<(Vec<f32>, f32)>,
    /// The subdomain's BaB depth (= number of split decisions).
    pub depth: usize,
    /// Wall-clock deadline for the whole verification (the oracle budgets its
    /// own slice inside this).
    pub deadline: Option<Instant>,
}

/// The oracle's verdict for one subdomain. See the module-level soundness
/// contract for when each variant is permitted.
#[derive(Debug)]
pub enum GraphMipLeafVerdict {
    /// Every requested row is certified-UNSAT on the subdomain: the child is
    /// verified and needs no further BaB.
    VerifiedAllRows,
    /// A CONFIRMED in-box counterexample (already revalidated through the
    /// graph forward by the oracle). `witness` is the flattened input point,
    /// `output` the revalidated graph output.
    Violated {
        /// Flattened input witness (inside the domain's input box).
        witness: Vec<f32>,
        /// The graph's output at `witness` (independent forward).
        output: Vec<f32>,
    },
    /// No usable verdict — the child continues in BaB unchanged.
    Undecided,
}

/// The leaf-solving oracle. Implementations live outside ny-propagate (for
/// example the ny-cli Graph-MIP encoder + ny-mip/ay solver); the signatures
/// are infallible by design — implementors map every internal failure to
/// [`GraphMipLeafVerdict::Undecided`].
pub trait GraphMipLeafOracle: Send + Sync {
    /// Attempt to certify all objective rows directly on an exact input-box
    /// leaf, before the Graph-MIP path collects intermediate node bounds.
    ///
    /// This default is intentionally fail-closed and source-compatible for
    /// every existing Graph-MIP-only oracle.  Implementations that opt in must
    /// enforce the [`GraphInputLeafRequest`] all-rows contract and their own
    /// bounded resource policy; grouped per-clause proofs are not sufficient
    /// without a future typed certified-row mask.
    fn solve_input_leaf(&self, _req: &GraphInputLeafRequest<'_>) -> GraphMipLeafVerdict {
        GraphMipLeafVerdict::Undecided
    }

    /// Attempt to decide one subdomain's undecided rows exactly. Must honor
    /// its own budget policy (eligibility gates + time slices) and the
    /// module-level soundness contract.
    fn solve_leaf(&self, req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict;

    /// May a `Violated` witness from this oracle be PUBLISHED as the run's sat
    /// candidate (#mip-leaf-witness)?
    ///
    /// FAIL-CLOSED DEFAULT: `false`. Publication requires an implementation to
    /// affirmatively state that the property's clause layout makes the lane's
    /// all-rows check sufficient, and only the caller that can SEE the layout
    /// can state it.
    ///
    /// Why this cannot be decided in this crate: the lane receives objectives
    /// already FLATTENED (`build_multi_objectives` walks
    /// `VnnLibSpec::output_constraints` and discards clause boundaries), and
    /// `ny-propagate` has no knowledge of `per_clause_input_bounds` at all. On
    /// an OR-of-AND property whose clauses carry their OWN input boxes, every
    /// output row holding at one point does NOT imply any clause holds, so the
    /// all-rows rule is NOT sufficient there — the honest statement of its
    /// scope lives on `witness_violates_every_objective_row`.
    ///
    /// Returning `false` costs at most a `sat` this lane would have had to
    /// guess at; the child is requeued and the search continues, exactly as
    /// before the witness plumbing existed.
    fn may_publish_violation_witness(&self) -> bool {
        false
    }
}

// ===========================================================================
// Root-bounds stash: the per-property α-CROWN node bounds, for MIP reuse
// ===========================================================================

fn graph_mip_consumer_enabled(value: Option<&str>) -> bool {
    value != Some("0")
}

fn graph_mip_stash_enabled_from_value(
    whole_net: Option<&str>,
    phase_budget: &PhaseBudgetConfig,
) -> bool {
    graph_mip_consumer_enabled(whole_net) && phase_budget.requests_mip_reservation()
}

/// Whether the default-on whole-network Graph-MIP consumer is armed and owns a
/// nonzero phase reservation. The leaf oracle receives its node bounds directly
/// in [`GraphMipLeafRequest`] and never reads this mailbox, so leaf-only and
/// zero-reservation runs must not clone a large unused map.
fn graph_mip_stash_enabled(phase_budget: &PhaseBudgetConfig) -> bool {
    let whole_net = std::env::var("NY_GRAPH_MIP").ok();
    graph_mip_stash_enabled_from_value(whole_net.as_deref(), phase_budget)
}

#[derive(Debug, PartialEq, Eq)]
struct RootBoundsStashKey {
    graph_scope: crate::beta_crown::bab_cuts::CutFoldScope,
    input_identity: Box<[u8]>,
}

fn append_usize_identity(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn append_array_identity(out: &mut Vec<u8>, array: &ndarray::ArrayD<f32>) {
    append_usize_identity(out, array.ndim());
    for &dim in array.shape() {
        append_usize_identity(out, dim);
    }
    append_usize_identity(out, array.len());
    for &value in array {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

/// Collision-proof identity for the exact model/input pair whose root bounds
/// are stashed. This is a one-slot mailbox, so hashing buys nothing: compare
/// the versioned direct encoding instead. Shape and every lower/upper f32 bit
/// are included, as is the optional L2 annotation because it can legitimately
/// tighten the resulting map. `graph_scope` distinguishes independently built
/// same-shaped models and survives only semantically identical clones.
fn root_bounds_stash_key(graph: &GraphNetwork, input: &BoundedTensor) -> RootBoundsStashKey {
    let mut input_identity = Vec::with_capacity(input.len().saturating_mul(8).saturating_add(128));
    input_identity.extend_from_slice(b"NY_GRAPH_MIP_ROOT_BOUNDS_STASH_V1\0");
    append_array_identity(&mut input_identity, input.lower());
    append_array_identity(&mut input_identity, input.upper());
    match input.l2_constraint() {
        Some(l2) => {
            input_identity.push(1);
            append_usize_identity(&mut input_identity, l2.axis());
            append_array_identity(&mut input_identity, l2.center());
            append_array_identity(&mut input_identity, l2.radius());
        }
        None => input_identity.push(0),
    }
    RootBoundsStashKey {
        graph_scope: graph.cut_fold_scope(),
        input_identity: input_identity.into_boxed_slice(),
    }
}

struct RootBoundsStashEntry {
    key: RootBoundsStashKey,
    bounds: Arc<HashMap<String, BoundedTensor>>,
}

thread_local! {
    /// One-slot mailbox for the per-property root node bounds. FILLED by the
    /// bound-producing sites — the ny-cli per-constraint precompute
    /// (`verify/graph.rs`) AND ny-propagate's own BaB bootstrap freeze points
    /// (the multi-objective root evaluation and the single-objective ReLU-split
    /// loop), which is where cifar100's relational/multi-clause lane actually
    /// computes them. READ by the ny-cli Graph-MIP escalation so it reuses the
    /// full-budget α-CROWN map instead of a deadline-truncated recompute.
    /// Same-thread by construction (bounds phase and escalation run
    /// synchronously on the verify thread); last writer wins and the exact
    /// model/input key rejects stale entries.
    static ROOT_BOUNDS_STASH: std::cell::RefCell<Option<RootBoundsStashEntry>> =
        const { std::cell::RefCell::new(None) };
}

/// Stash the per-property root node bounds for a later Graph-MIP consumer.
/// The mailbox remains bounded to one map and is disabled when
/// `NY_GRAPH_MIP=0` or the phase policy reserves no whole-network MIP time;
/// the leaf oracle does not consume it.
pub fn stash_root_bounds_for_mip(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    phase_budget: &PhaseBudgetConfig,
    node_bounds: &HashMap<String, BoundedTensor>,
) {
    update_root_bounds_stash(
        graph,
        input,
        node_bounds,
        graph_mip_stash_enabled(phase_budget),
    );
}

/// Apply an already-resolved stash admission decision. Keeping this helper
/// independent of process environment makes the clear/accept lifecycle
/// regression-testable without racing other tests over `NY_GRAPH_MIP`.
fn update_root_bounds_stash(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    enabled: bool,
) {
    if !enabled {
        ROOT_BOUNDS_STASH.with(|slot| *slot.borrow_mut() = None);
        return;
    }
    let key = root_bounds_stash_key(graph, input);
    let arc = Arc::new(node_bounds.clone());
    tracing::debug!(
        nodes = arc.len(),
        "Graph-MIP: stashed per-property root node bounds"
    );
    ROOT_BOUNDS_STASH.with(|s| {
        *s.borrow_mut() = Some(RootBoundsStashEntry { key, bounds: arc });
    });
}

/// Consume the stashed bounds when they were computed for exactly this model
/// and input. A mismatch also drops the stale one-slot entry.
pub fn stashed_root_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> Option<Arc<HashMap<String, BoundedTensor>>> {
    let key = root_bounds_stash_key(graph, input);
    ROOT_BOUNDS_STASH.with(|slot| {
        let entry = slot.borrow_mut().take()?;
        (entry.key == key).then_some(entry.bounds)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        graph_mip_stash_enabled_from_value, root_bounds_stash_key, update_root_bounds_stash,
        RootBoundsStashEntry, ROOT_BOUNDS_STASH,
    };
    use crate::layers::ReLULayer;
    use crate::{GraphNetwork, GraphNode, Layer, PhaseBudgetConfig};
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::{BoundedTensor, L2Constraint};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn input_with_shape(shape: &[usize], lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).expect("lower shape"),
            ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).expect("upper shape"),
        )
        .expect("valid input box")
    }

    #[test]
    fn graph_mip_stash_gate_tracks_consumer_and_reservation_policy() {
        let default_policy = PhaseBudgetConfig::default();
        assert!(graph_mip_stash_enabled_from_value(None, &default_policy));
        assert!(graph_mip_stash_enabled_from_value(
            Some("1"),
            &default_policy
        ));
        assert!(
            !graph_mip_stash_enabled_from_value(Some("0"), &default_policy),
            "whole-network Graph-MIP's exact-zero kill switch disables its stash"
        );

        let zero_policy = PhaseBudgetConfig {
            mip_min_fraction: 0.0,
            mip_min_secs: 0,
            ..Default::default()
        };
        assert!(
            !graph_mip_stash_enabled_from_value(None, &zero_policy),
            "default-on whole-net MIP must not clone bounds for a zero reservation"
        );
        assert!(
            !graph_mip_stash_enabled_from_value(Some("1"), &zero_policy),
            "an env-enabled consumer still needs an admitted nonzero reservation"
        );
    }

    #[test]
    fn root_bounds_stash_accepts_admitted_policy_and_clears_on_decline() {
        let graph = GraphNetwork::new();
        let input = input_with_shape(&[1], &[-1.0], &[1.0]);
        let node_bounds = HashMap::from([("root".to_string(), input.clone())]);

        update_root_bounds_stash(&graph, &input, &node_bounds, true);
        let accepted = super::stashed_root_bounds(&graph, &input).expect("admitted stash");
        assert_eq!(accepted.len(), 1);

        update_root_bounds_stash(&graph, &input, &node_bounds, true);
        update_root_bounds_stash(&graph, &input, &node_bounds, false);
        assert!(
            super::stashed_root_bounds(&graph, &input).is_none(),
            "a declined producer must clear a stale one-slot stash"
        );
    }

    #[test]
    fn root_bounds_stash_key_is_exact_for_graph_box_shape_and_l2() {
        let graph = GraphNetwork::new();
        let input = input_with_shape(&[2], &[-1.0, -0.5], &[1.0, 0.5]);
        let base = root_bounds_stash_key(&graph, &input);

        let identical_clone = graph.clone();
        assert_eq!(
            base,
            root_bounds_stash_key(&identical_clone, &input),
            "a semantically identical graph clone must retain its scope"
        );
        assert_ne!(
            base,
            root_bounds_stash_key(&GraphNetwork::new(), &input),
            "an independently built same-shaped graph must miss"
        );

        let lower_ulp = input_with_shape(
            &[2],
            &[f32::from_bits((-1.0_f32).to_bits() + 1), -0.5],
            &[1.0, 0.5],
        );
        assert_ne!(base, root_bounds_stash_key(&graph, &lower_ulp));
        let upper_ulp = input_with_shape(
            &[2],
            &[-1.0, -0.5],
            &[f32::from_bits(1.0_f32.to_bits() + 1), 0.5],
        );
        assert_ne!(base, root_bounds_stash_key(&graph, &upper_ulp));

        let reshaped = input_with_shape(&[1, 2], &[-1.0, -0.5], &[1.0, 0.5]);
        assert_ne!(
            base,
            root_bounds_stash_key(&graph, &reshaped),
            "equal flattened bits with a different shape must miss"
        );

        let l2 = L2Constraint::new(
            ArrayD::zeros(IxDyn(&[2])),
            ArrayD::from_elem(IxDyn(&[]), 1.0),
            0,
            &[2],
        )
        .expect("valid rank-one L2 constraint");
        let annotated = input.clone().with_l2_constraint(l2);
        assert_ne!(
            base,
            root_bounds_stash_key(&graph, &annotated),
            "an L2 tightening annotation is part of exact input identity"
        );

        let mut retargeted = graph.clone();
        retargeted.set_output("different-output");
        assert_ne!(
            base,
            root_bounds_stash_key(&retargeted, &input),
            "a semantic graph mutation must mint a new scope"
        );

        let mut extended = graph;
        extended.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        assert_ne!(
            base,
            root_bounds_stash_key(&extended, &input),
            "a structural graph mutation must mint a new scope"
        );
    }

    #[test]
    fn root_bounds_stash_is_one_shot_and_rejects_foreign_graphs() {
        let graph = GraphNetwork::new();
        let input = input_with_shape(&[1], &[-1.0], &[1.0]);
        ROOT_BOUNDS_STASH.with(|slot| {
            *slot.borrow_mut() = Some(RootBoundsStashEntry {
                key: root_bounds_stash_key(&graph, &input),
                bounds: Arc::new(HashMap::new()),
            });
        });
        assert!(super::stashed_root_bounds(&GraphNetwork::new(), &input).is_none());
        assert!(
            super::stashed_root_bounds(&graph, &input).is_none(),
            "a mismatched fetch consumes the stale one-slot entry"
        );

        ROOT_BOUNDS_STASH.with(|slot| {
            *slot.borrow_mut() = Some(RootBoundsStashEntry {
                key: root_bounds_stash_key(&graph, &input),
                bounds: Arc::new(HashMap::new()),
            });
        });
        assert!(super::stashed_root_bounds(&graph, &input).is_some());
        assert!(
            super::stashed_root_bounds(&graph, &input).is_none(),
            "a successful fetch consumes the entry"
        );
    }
}
