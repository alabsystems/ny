// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Root evaluation and root-domain assembly for multi-objective graph BaB.
//!
//! Keeps the top-level coordinator in `verify.rs` focused on queue flow while
//! preserving the existing root semantics, including the `#3813` cached-lA
//! warm-start path.

use ny_core::{GemmEngine, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::{debug, info};

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::domain::{MultiObjectiveGraphBabDomain, ObjectiveAggregation};
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::beta_crown::state::GraphDomainAlphaState;
use crate::bounds::{AdamParams, AlphaCrownConfig};
use crate::network::{
    root_alpha_cuda_margin_step_enabled, root_alpha_cuda_rows_enabled, AtomicCudaMarginStepCommit,
    AtomicCudaMarginStepOutcome, AtomicCudaMarginStepRequest, AtomicCudaRowsCommit,
    AtomicCudaRowsOutcome, AtomicCudaRowsRefusal, AtomicCudaRowsRequest, SpecCrownRequest,
};
use crate::{BetaCrownConfig, GraphNetwork, OwnedSignNormalizedObjectiveSet};

use super::super::super::BetaCrownVerifier;
use super::super::shared::init::{
    compute_graph_bab_bootstrap, compute_graph_bab_bootstrap_with_phase_cap_checkpoint,
    compute_graph_root_output_bounds, GraphBabBootstrap,
};
use super::super::shared::setup::{
    build_graph_bab_setup, build_graph_bab_setup_owned, build_graph_cut_pool,
    build_initial_node_bounds_arc, build_root_alpha_state, GraphBabSetup,
};
use super::active_set_gpu_alpha::{
    active_set_full_state_identity, classify_active_set_gpu_alpha,
    run_active_set_gpu_alpha_lr_bracket, ActiveSetCertifiedPairFingerprint,
    ActiveSetGpuAlphaCandidateTrace, ActiveSetGpuAlphaClassification,
    ActiveSetGpuAlphaExecutionOutput, ActiveSetGpuAlphaExecutionRefusal,
    ActiveSetGpuAlphaFullStateIdentity, ActiveSetGpuAlphaPlan, ActiveSetGpuAlphaRefusal,
    ActiveSetGpuAlphaScore, ActiveSetGpuAlphaSelectedCandidate, ActiveSetUnresolvedRow,
};
use super::critical_gpu_alpha::{
    alpha_state_identity, critical_gpu_alpha_lr_bracket_enabled, fingerprint_bytes,
    fingerprint_u64, run_critical_gpu_alpha_lr_bracket, run_one_critical_gpu_alpha_step,
    CriticalGpuAlphaCandidateTrace, CriticalGpuAlphaCertifiedPair, CriticalGpuAlphaStateIdentity,
    CriticalGpuAlphaStepOutput, CriticalGpuAlphaStepRefusal,
};
use super::dd_zono_root::{intersect_objective_bounds, run_dd_zono_root};
use super::output_conditioned_head::try_output_conditioned_root_refutation;

/// Intersect the certified zonotope's per-node enclosures into the stored bounds
/// map (`#dd-zono-interm`). Returns how many nodes were tightened.
///
/// SHRINK-ONLY, on the same terms as `stabilize_and_fix`: for each element the
/// published bound is `max(stored_l, zono_l)` / `min(stored_u, zono_u)`. Both
/// operands are sound enclosures of the same node over the same input box, so the
/// intersection is a sound enclosure and can only be tighter.
///
/// Every degenerate case is REFUSED rather than published: arity mismatch, a
/// non-finite candidate, or an intersection that would cross (`l > u`, which would
/// describe an empty set and could license anything downstream). A refused node
/// keeps its stored bound untouched, so a refusal is byte-identical to today.
pub(super) fn intersect_interm_into_stored(
    stored: &mut std::collections::HashMap<String, BoundedTensor>,
    zono: &std::collections::HashMap<String, (Vec<f64>, Vec<f64>)>,
) -> usize {
    let mut tightened = 0usize;
    for (name, (zl, zu)) in zono {
        let Some(current) = stored.get(name) else {
            continue;
        };
        let cl = current.lower();
        let cu = current.upper();
        if cl.len() != zl.len() || cu.len() != zu.len() {
            continue;
        }
        let mut new_l = cl.clone();
        let mut new_u = cu.clone();
        let mut changed = false;
        let mut refuse = false;
        for (i, (lo, up)) in new_l.iter_mut().zip(new_u.iter_mut()).enumerate() {
            // Narrow OUTWARD on the f64 -> f32 cast so the published bound can
            // never be tighter than what was certified.
            let zlo = next_down_f32(zl[i] as f32);
            let zup = next_up_f32(zu[i] as f32);
            if !zlo.is_finite() || !zup.is_finite() || zlo > zup {
                refuse = true;
                break;
            }
            let cand_l = lo.max(zlo);
            let cand_u = up.min(zup);
            if cand_l > cand_u {
                refuse = true;
                break;
            }
            if cand_l > *lo || cand_u < *up {
                changed = true;
            }
            *lo = cand_l;
            *up = cand_u;
        }
        if refuse || !changed {
            continue;
        }
        if let Ok(bt) = BoundedTensor::new(new_l, new_u) {
            stored.insert(name.clone(), bt);
            tightened += 1;
        }
    }
    tightened
}

#[cfg(test)]
mod dd_zono_interm_tests {
    use std::collections::HashMap;

    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    use super::intersect_interm_into_stored;

    fn bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn certified_intermediate_intersection_is_shrink_only() {
        let mut stored = HashMap::from([("relu".to_string(), bounds(&[-2.0, -1.0], &[3.0, 4.0]))]);
        let zono = HashMap::from([("relu".to_string(), (vec![-0.5, -9.0], vec![1.5, 9.0]))]);

        assert_eq!(intersect_interm_into_stored(&mut stored, &zono), 1);
        let tightened = &stored["relu"];
        assert!(tightened.lower()[0] <= -0.5 && tightened.lower()[0] > -0.51);
        assert!(tightened.upper()[0] >= 1.5 && tightened.upper()[0] < 1.51);
        assert_eq!(tightened.lower()[1], -1.0);
        assert_eq!(tightened.upper()[1], 4.0);
    }

    #[test]
    fn malformed_or_disjoint_intermediate_is_refused_atomically() {
        let original = bounds(&[-2.0, -1.0], &[3.0, 4.0]);
        for candidate in [
            (vec![f64::NAN, 0.0], vec![1.0, 1.0]),
            (vec![8.0, 0.0], vec![9.0, 1.0]),
            (vec![0.0], vec![1.0]),
        ] {
            let mut stored = HashMap::from([("relu".to_string(), original.clone())]);
            let zono = HashMap::from([("relu".to_string(), candidate)]);
            assert_eq!(intersect_interm_into_stored(&mut stored, &zono), 0);
            assert_eq!(stored["relu"].lower(), original.lower());
            assert_eq!(stored["relu"].upper(), original.upper());
        }
    }
}
use super::per_disjunct::build_per_disjunct_alphas;
use super::post_c_survivor::{
    build_post_c_survivor_plan, run_post_c_survivor_candidate, PostCSurvivorAccepted,
};
use super::selective_root_alpha::{sound_shared_gpu_available, SelectiveRootAlphaGate};
use super::shared::{build_spec_matrix, spec_bounds_to_vec};

pub(super) enum MultiObjectiveRootOutcome {
    Finished(Box<BetaCrownResult>),
    Continue(Box<MultiObjectiveRootState>),
}

/// One sign-normalized property source retained across root evaluation.
///
/// The borrowed variant preserves every established public verifier call. The
/// owned variant is reachable only through the explicit consuming ingress and
/// keeps the original non-`Clone` allocation owner intact. Callers may take
/// short immutable views, but cannot split an owned source into independently
/// movable row and threshold carriers.
#[must_use]
pub(super) enum MultiObjectiveProperty<'a> {
    Borrowed {
        objectives: &'a [Vec<f32>],
        thresholds: &'a [f32],
    },
    Owned(OwnedSignNormalizedObjectiveSet),
}

impl<'a> MultiObjectiveProperty<'a> {
    #[inline]
    pub(super) fn borrowed(objectives: &'a [Vec<f32>], thresholds: &'a [f32]) -> Self {
        Self::Borrowed {
            objectives,
            thresholds,
        }
    }

    #[inline]
    pub(super) fn owned(source: OwnedSignNormalizedObjectiveSet) -> Self {
        Self::Owned(source)
    }

    #[inline]
    pub(super) fn views(&self) -> (&[Vec<f32>], &[f32]) {
        match self {
            Self::Borrowed {
                objectives,
                thresholds,
            } => (objectives, thresholds),
            Self::Owned(objectives) => (objectives.rows(), objectives.thresholds()),
        }
    }
}

#[must_use]
pub(super) struct MultiObjectiveRootState {
    /// The exact output enclosure produced by `RootObjectiveEvaluation`.
    ///
    /// Root-terminal paths preserve their established result behavior. A
    /// continuing path retains the original allocation for the finalized-root
    /// custody boundary instead of dropping it at the end of root evaluation.
    pub(super) initial_output: BoundedTensor,
    pub(super) root_domain: MultiObjectiveGraphBabDomain,
    /// Correctly expanded warmup W, retained separately from established H.
    ///
    /// This is one-shot state for the children split directly from the root in
    /// executable shared batch zero; it is never transported to a later wave
    /// or to descendants of an already split child.
    pub(super) selective_root_alpha_candidate: Option<GraphDomainAlphaState>,
    pub(super) relu_nodes: Vec<String>,
    pub(super) cut_pool: GraphCutPool,
    pub(super) use_batched_gpu: bool,
}

/// Root borrows and returned-property custody have deliberately distinct
/// lifetimes. In particular, a borrowed property never prolongs the temporary
/// borrow of the configured graph, which must be movable after root evaluation.
#[must_use]
pub(super) struct MultiObjectiveRootRequest<'context, 'property> {
    pub(super) verifier: &'context BetaCrownVerifier,
    pub(super) graph: &'context GraphNetwork,
    pub(super) input: &'context BoundedTensor,
    pub(super) property: MultiObjectiveProperty<'property>,
    pub(super) engine: Option<&'context dyn GemmEngine>,
    pub(super) conjunctive: bool,
    pub(super) deadline: Option<std::time::Instant>,
}

/// Root result paired with the exact property value that produced it.
///
/// Returning the property for both terminal and continuing outcomes makes its
/// custody independent of root control flow. The outer verifier may keep using
/// borrowed views for ordinary BaB, while a terminal result simply drops the
/// still-intact owner after root evaluation has completed.
#[must_use]
pub(super) struct MultiObjectiveRootEvaluation<'property> {
    pub(super) outcome: MultiObjectiveRootOutcome,
    pub(super) property: MultiObjectiveProperty<'property>,
}

#[must_use]
struct BorrowedMultiObjectiveRootRequest<'context, 'property_view> {
    verifier: &'context BetaCrownVerifier,
    graph: &'context GraphNetwork,
    input: &'context BoundedTensor,
    objectives: &'property_view [Vec<f32>],
    thresholds: &'property_view [f32],
    engine: Option<&'context dyn GemmEngine>,
    conjunctive: bool,
    deadline: Option<std::time::Instant>,
}

#[cfg(test)]
mod owned_property_custody_tests {
    use std::time::{Duration, Instant};

    use ndarray::{arr1, arr2};

    use super::*;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::GraphNode;

    struct AllocationIdentity {
        outer_pointer: *const Vec<f32>,
        outer_capacity: usize,
        row_pointers: [*const f32; 2],
        row_capacities: [usize; 2],
        threshold_pointer: *const f32,
        threshold_capacity: usize,
    }

    fn vec_with_capacity(values: &[f32], capacity: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(capacity);
        result.extend_from_slice(values);
        result
    }

    fn owned_property(threshold: f32) -> (OwnedSignNormalizedObjectiveSet, AllocationIdentity) {
        let first = vec_with_capacity(&[1.0], 11);
        let second = vec_with_capacity(&[1.0], 13);
        let row_pointers = [first.as_ptr(), second.as_ptr()];
        let row_capacities = [first.capacity(), second.capacity()];
        let mut rows = Vec::with_capacity(7);
        rows.extend([first, second]);
        let outer_pointer = rows.as_ptr();
        let outer_capacity = rows.capacity();

        let thresholds = vec_with_capacity(&[threshold, threshold], 17);
        let threshold_pointer = thresholds.as_ptr();
        let threshold_capacity = thresholds.capacity();
        let owner = OwnedSignNormalizedObjectiveSet::new(rows, thresholds);

        (
            owner,
            AllocationIdentity {
                outer_pointer,
                outer_capacity,
                row_pointers,
                row_capacities,
                threshold_pointer,
                threshold_capacity,
            },
        )
    }

    fn assert_same_allocation(
        property: &MultiObjectiveProperty<'_>,
        expected: &AllocationIdentity,
    ) {
        let owner = match property {
            MultiObjectiveProperty::Owned(owner) => owner,
            MultiObjectiveProperty::Borrowed { .. } => {
                panic!("consuming root request returned a borrowed property")
            }
        };
        let (outer_pointer, outer_capacity, threshold_pointer, threshold_capacity) =
            owner.allocation_custody_for_test();
        assert_eq!(outer_pointer, expected.outer_pointer);
        assert_eq!(outer_capacity, expected.outer_capacity);
        assert_eq!(owner.rows()[0].as_ptr(), expected.row_pointers[0]);
        assert_eq!(owner.rows()[1].as_ptr(), expected.row_pointers[1]);
        assert_eq!(owner.rows()[0].capacity(), expected.row_capacities[0]);
        assert_eq!(owner.rows()[1].capacity(), expected.row_capacities[1]);
        assert_eq!(threshold_pointer, expected.threshold_pointer);
        assert_eq!(threshold_capacity, expected.threshold_capacity);
    }

    fn branchy_identity_graph() -> (GraphNetwork, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("linear")),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["linear".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("output")),
            vec!["relu".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("bounded input");
        (graph, input)
    }

    fn verifier() -> BetaCrownVerifier {
        BetaCrownVerifier::new(BetaCrownConfig {
            timeout: Duration::from_secs(5),
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_pgd_attack: false,
            enable_cuts: false,
            batch_size: 1,
            ..Default::default()
        })
    }

    fn evaluate_owned_root(
        threshold: f32,
        deadline: Option<Instant>,
        inspect: impl FnOnce(&MultiObjectiveRootOutcome),
    ) {
        let (source_graph, input) = branchy_identity_graph();
        let verifier = verifier();
        let graph = verifier.configured_graph_for_crown(&source_graph);
        let (owner, allocation) = owned_property(threshold);
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());

        let evaluation = evaluate_root(
            MultiObjectiveRootRequest {
                verifier: &verifier,
                graph: &graph,
                input: &input,
                property: MultiObjectiveProperty::owned(owner),
                engine: None,
                conjunctive: false,
                deadline,
            },
            &mut lifecycle,
        )
        .expect("toy root evaluation");

        inspect(&evaluation.outcome);
        assert_same_allocation(&evaluation.property, &allocation);
    }

    #[ntest::timeout(15000)]
    #[test]
    fn owned_allocation_survives_continuing_root_evaluation() {
        evaluate_owned_root(0.5, None, |outcome| {
            assert!(matches!(outcome, MultiObjectiveRootOutcome::Continue(_)));
        });
    }

    #[ntest::timeout(15000)]
    #[test]
    fn owned_allocation_survives_terminal_timeout_root_evaluation() {
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("Instant subtraction");
        evaluate_owned_root(0.5, Some(expired), |outcome| match outcome {
            MultiObjectiveRootOutcome::Finished(result) => assert!(matches!(
                result.result,
                BabVerificationStatus::Timeout | BabVerificationStatus::Unknown { .. }
            )),
            MultiObjectiveRootOutcome::Continue(_) => {
                panic!("expired root authority must be terminal")
            }
        });
    }
}

#[must_use]
pub(super) struct RootObjectiveEvaluation {
    pub(super) initial_output: BoundedTensor,
    pub(super) initial_obj_bounds: Vec<(f32, f32)>,
    pub(super) root_spec_cache: Option<CachedLinearBounds>,
    /// Exact-dark override produced only when the complete alpha1 `C` vector
    /// was evaluated and won whole-candidate selection.
    pub(super) root_alpha_override: Option<crate::bounds::GraphAlphaState>,
    /// Full-objective indices represented by `root_spec_cache`'s rows.
    ///
    /// The default/full-spec path stores `0..objectives.len()`.  The dark
    /// `NY_ROOT_SPEC_PRUNE=1` path stores only still-active rows; attachment
    /// expands them back to a full `Vec<Option<_>>`, leaving certified-pruned
    /// objectives as `None` so a cache can never be applied to the wrong row.
    pub(super) root_spec_cache_active_indices: Vec<usize>,
}

/// A sound pre-CROWN compression plan for disjunctive root specifications.
///
/// `pre_bounds` comes exclusively from the bootstrap's certified full-output
/// enclosure.  Rows whose lower endpoint is already strictly above their
/// threshold need no optimized specification backward.  The output enclosure
/// remains valid even when a later root-tightening pass only shrank intermediate
/// boxes, so it is also the sound result source for an all-pruned plan.
#[derive(Debug)]
struct RootSpecPrunePlan {
    bootstrap_output: BoundedTensor,
    pre_bounds: Vec<(f32, f32)>,
    active_indices: Vec<usize>,
    active_spec_matrix: Option<ndarray::Array2<f32>>,
}

/// One validated compact view of the unresolved source rows for the typed
/// exact-C margin transaction.
///
/// Every compact position has one immutable source-row identity shared by its
/// objective, threshold, C row, and independently certified bootstrap
/// reference interval.  Keeping those values in one owned carrier prevents a
/// later caller from independently filtering only some of the row-bearing
/// inputs.
struct AtomicRootCCompactRows {
    source_indices: Vec<usize>,
    objectives: Vec<Vec<f32>>,
    thresholds: Vec<f32>,
    spec_matrix: ndarray::Array2<f32>,
    reference: BoundedTensor,
}

fn root_spec_prune_enabled() -> bool {
    std::env::var("NY_ROOT_SPEC_PRUNE").ok().as_deref() == Some("1")
}

/// Authority predicate for removing a row before optimized spec CROWN.
/// Every endpoint must be an ordinary, ordered finite enclosure; malformed or
/// unbounded intervals always stay active even if their lower endpoint alone
/// would compare above the threshold.
fn root_prebound_certifies(lower: f32, upper: f32, threshold: f32) -> bool {
    root_interval_is_finite_ordered(lower, upper) && threshold.is_finite() && lower > threshold
}

fn root_interval_is_finite_ordered(lower: f32, upper: f32) -> bool {
    lower.is_finite() && upper.is_finite() && lower <= upper
}

/// Resolve the exact output entry retained by the bootstrap.  Any graph-order
/// or lookup problem declines the optimization so the caller runs the historic
/// full-spec request unchanged.
fn bootstrap_output_bounds<'a>(
    graph: &GraphNetwork,
    bootstrap: &'a GraphBabBootstrap,
) -> Option<&'a BoundedTensor> {
    let output_name = if graph.output_name().is_empty() {
        graph.exec_order().ok()?.last()?.clone()
    } else {
        graph.output_name().to_string()
    };
    bootstrap.initial_node_bounds.get(&output_name)
}

/// Select the full-output enclosure after the atomic root-`C` transaction.
///
/// Every committed Stage-A route already authenticated the bootstrap output and
/// used its objective projection as the independent reference enclosure.  The
/// complete-`C` result, not this full-output tensor, is root-verdict authority.
/// Re-running plain IBP after commitment is therefore redundant and can turn a
/// successful transaction into a timeout: exact-`C` may legitimately consume
/// its local root deadline, leaving no time for that second forward pass.
///
/// Keep both inputs lazy so the historic non-atomic path neither resolves the
/// bootstrap output nor changes its existing IBP request.  Conversely, a
/// committed route never calls the fallback.  A missing bootstrap output after
/// commitment is an invariant violation and fails closed rather than crossing
/// into a second backend transaction.
fn resolve_root_output_after_atomic_stage_a<B, F>(
    atomic_stage_a_committed: bool,
    authenticated_bootstrap_output: B,
    ordinary_output: F,
) -> Result<BoundedTensor>
where
    B: FnOnce() -> Option<BoundedTensor>,
    F: FnOnce() -> Result<BoundedTensor>,
{
    if atomic_stage_a_committed {
        return authenticated_bootstrap_output().ok_or_else(|| {
            ny_core::NyError::InvalidSpec(
                "atomic root-C committed without its authenticated bootstrap output".to_string(),
            )
        });
    }
    ordinary_output()
}

/// Graph/bootstrap integration seam used by the root-objective caller.
/// Keeping the real output-name lookup here makes the committed-path test
/// exercise the same graph identity and bootstrap map as production, while
/// [`resolve_root_output_after_atomic_stage_a`] retains the small lazy branch
/// truth table.
fn resolve_graph_root_output_after_atomic_stage_a<F>(
    graph: &GraphNetwork,
    bootstrap: &GraphBabBootstrap,
    atomic_stage_a_committed: bool,
    ordinary_output: F,
) -> Result<BoundedTensor>
where
    F: FnOnce() -> Result<BoundedTensor>,
{
    resolve_root_output_after_atomic_stage_a(
        atomic_stage_a_committed,
        || bootstrap_output_bounds(graph, bootstrap).cloned(),
        ordinary_output,
    )
}

#[cfg(test)]
mod atomic_stage_a_root_output_tests {
    use super::{
        resolve_graph_root_output_after_atomic_stage_a, resolve_root_output_after_atomic_stage_a,
    };
    use crate::beta_crown::config::BetaCrownConfig;
    use crate::beta_crown::engine::graph::shared::init::compute_graph_bab_bootstrap;
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer};
    use ndarray::{arr1, arr2};
    use ny_core::{NyError, Result};
    use ny_tensor::BoundedTensor;
    use std::cell::Cell;

    fn output(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
            .expect("ordered finite output")
    }

    fn endpoint_bits(output: &BoundedTensor) -> (u32, u32) {
        (output.lower()[[0]].to_bits(), output.upper()[[0]].to_bits())
    }

    #[test]
    fn committed_stage_a_reuses_authenticated_output_without_calling_fallback() {
        let authenticated = output(-1.25, 2.5);
        let expected = endpoint_bits(&authenticated);
        let authenticated_calls = Cell::new(0usize);
        let fallback_calls = Cell::new(0usize);

        let selected = resolve_root_output_after_atomic_stage_a(
            true,
            || {
                authenticated_calls.set(authenticated_calls.get() + 1);
                Some(authenticated.clone())
            },
            || -> Result<BoundedTensor> {
                fallback_calls.set(fallback_calls.get() + 1);
                Err(NyError::DeadlineExceeded(
                    "expired root deadline must not be consulted after commitment".to_string(),
                ))
            },
        )
        .expect("the authenticated bootstrap output remains a sound enclosure");

        assert_eq!(endpoint_bits(&selected), expected);
        assert_eq!(authenticated_calls.get(), 1);
        assert_eq!(fallback_calls.get(), 0);
    }

    #[test]
    fn ordinary_route_calls_only_the_historical_fallback() {
        let fallback = output(-3.0, 4.0);
        let expected = endpoint_bits(&fallback);
        let authenticated_calls = Cell::new(0usize);
        let fallback_calls = Cell::new(0usize);

        let selected = resolve_root_output_after_atomic_stage_a(
            false,
            || {
                authenticated_calls.set(authenticated_calls.get() + 1);
                Some(output(-1.0, 1.0))
            },
            || {
                fallback_calls.set(fallback_calls.get() + 1);
                Ok(fallback.clone())
            },
        )
        .expect("historical fallback result");

        assert_eq!(endpoint_bits(&selected), expected);
        assert_eq!(authenticated_calls.get(), 0);
        assert_eq!(fallback_calls.get(), 1);
    }

    #[test]
    fn committed_stage_a_without_authenticated_output_fails_closed() {
        let fallback_calls = Cell::new(0usize);
        let error = resolve_root_output_after_atomic_stage_a(
            true,
            || None,
            || {
                fallback_calls.set(fallback_calls.get() + 1);
                Ok(output(-1.0, 1.0))
            },
        )
        .expect_err("commitment cannot cross into a second backend transaction");

        assert!(matches!(error, NyError::InvalidSpec(_)));
        assert_eq!(fallback_calls.get(), 0);
    }

    #[test]
    fn committed_graph_route_uses_the_real_bootstrap_output_identity_and_shape() {
        let mut graph = GraphNetwork::new();
        let output_layer = LinearLayer::new(
            arr2(&[[2.0_f32, -1.0], [0.5, 3.0]]),
            Some(arr1(&[0.25_f32, -0.5])),
        )
        .expect("valid output layer");
        graph.add_node(GraphNode::from_input(
            "sealed_output",
            Layer::Linear(output_layer),
        ));
        graph.set_output("sealed_output");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.25]).into_dyn(),
            arr1(&[2.0_f32, 1.5]).into_dyn(),
        )
        .expect("valid input enclosure");
        let config = BetaCrownConfig {
            use_alpha_crown: false,
            ..BetaCrownConfig::default()
        };
        let bootstrap = compute_graph_bab_bootstrap(&graph, &input, &config, None, None)
            .expect("real graph bootstrap");
        let expected = bootstrap
            .initial_node_bounds
            .get("sealed_output")
            .expect("bootstrap output node");
        let expected_shape = expected.shape().to_vec();
        let expected_bits = expected
            .lower()
            .iter()
            .chain(expected.upper().iter())
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let fallback_calls = Cell::new(0usize);

        let selected = resolve_graph_root_output_after_atomic_stage_a(
            &graph,
            &bootstrap,
            true,
            || -> Result<BoundedTensor> {
                fallback_calls.set(fallback_calls.get() + 1);
                Err(NyError::DeadlineExceeded(
                    "expired post-commit fallback must stay lazy".to_string(),
                ))
            },
        )
        .expect("committed route reuses the graph's authenticated output");

        assert_eq!(selected.shape(), expected_shape.as_slice());
        assert_eq!(
            selected
                .lower()
                .iter()
                .chain(selected.upper().iter())
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_bits
        );
        assert_eq!(fallback_calls.get(), 0);
    }
}

fn graph_output_name(graph: &GraphNetwork) -> Option<String> {
    if graph.output_name().is_empty() {
        graph.exec_order().ok()?.last().cloned()
    } else {
        Some(graph.output_name().to_string())
    }
}

fn atomic_root_c_reference(
    output: &BoundedTensor,
    objectives: &[Vec<f32>],
) -> Option<BoundedTensor> {
    let projected = BetaCrownVerifier::objective_bounds_multi(output, objectives).ok()?;
    if projected.len() != objectives.len() {
        return None;
    }
    let (lower, upper): (Vec<f32>, Vec<f32>) = projected.into_iter().unzip();
    BoundedTensor::new(
        ndarray::Array1::from_vec(lower).into_dyn(),
        ndarray::Array1::from_vec(upper).into_dyn(),
    )
    .ok()
}

/// Score-bearing Stage-A candidate: evaluate the complete source-ordered root
/// `C` matrix at the bootstrap alpha state through one atomic factory
/// transaction. The independently projected bootstrap output is the committed
/// refusal authority.
fn run_atomic_root_c_stage_a(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    full_spec_matrix: &ndarray::Array2<f32>,
    bootstrap: &GraphBabBootstrap,
    authority_deadline: Option<std::time::Instant>,
) -> AtomicCudaRowsOutcome {
    let refuse = |refusal| AtomicCudaRowsOutcome::RefusedBeforeCommit { refusal };
    let Some(deadline) = authority_deadline else {
        return refuse(AtomicCudaRowsRefusal::MissingDeadline);
    };
    let Some(output_name) = graph_output_name(graph) else {
        return refuse(AtomicCudaRowsRefusal::MissingOutputNode);
    };
    let Some(output) = bootstrap.initial_node_bounds.get(&output_name) else {
        return refuse(AtomicCudaRowsRefusal::MissingReferenceOutput);
    };
    let Some(reference) = atomic_root_c_reference(output, objectives) else {
        return refuse(AtomicCudaRowsRefusal::ReferenceNonFiniteOrInverted);
    };
    let Some(alpha_state) = bootstrap.root_alpha_state.as_ref() else {
        return refuse(AtomicCudaRowsRefusal::MissingAlphaState);
    };

    AtomicCudaRowsRequest::new(
        graph,
        input,
        &output_name,
        &bootstrap.initial_node_bounds,
        alpha_state,
        full_spec_matrix,
        &reference,
        deadline,
    )
    .run()
}

struct AtomicRootCMarginAlphaSchedule {
    /// Historical/default-dark one-step LR: the next point on the completed
    /// alpha-CROWN exponential schedule.
    adam: AdamParams,
    /// Continue the configured alpha-CROWN schedule after every accepted or
    /// rejected proposal attempt; the optimizer time index advances with it.
    learning_rate_decay: f32,
    /// The independently gated LR bracket is defined around the configured
    /// category base, not the already-decayed historical one-step LR.
    bracket_base_learning_rate: f32,
}

fn atomic_root_c_margin_alpha_schedule(
    alpha_config: &AlphaCrownConfig,
    optimizer_updates_override: Option<usize>,
) -> AtomicRootCMarginAlphaSchedule {
    let completed_iters = optimizer_updates_override.unwrap_or(alpha_config.iterations);
    let next_decayed_learning_rate =
        alpha_config.learning_rate * alpha_config.lr_decay.powf(completed_iters as f32);
    AtomicRootCMarginAlphaSchedule {
        adam: alpha_config.adam_params(
            next_decayed_learning_rate,
            completed_iters.saturating_add(1),
        ),
        learning_rate_decay: alpha_config.lr_decay,
        bracket_base_learning_rate: alpha_config.learning_rate,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_atomic_root_c_margin_step(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    spec_matrix: &ndarray::Array2<f32>,
    reference_override: Option<&BoundedTensor>,
    bootstrap: &GraphBabBootstrap,
    conjunctive: bool,
    verify_upper_bound: bool,
    multi_iterations: usize,
    authority_deadline: Option<std::time::Instant>,
) -> AtomicCudaMarginStepOutcome {
    let refuse = |refusal| AtomicCudaMarginStepOutcome::RefusedBeforeCommit { refusal };
    let Some(deadline) = authority_deadline else {
        return refuse(AtomicCudaRowsRefusal::MissingDeadline);
    };
    let Some(output_name) = graph_output_name(graph) else {
        return refuse(AtomicCudaRowsRefusal::MissingOutputNode);
    };
    let Some(output) = bootstrap.initial_node_bounds.get(&output_name) else {
        return refuse(AtomicCudaRowsRefusal::MissingReferenceOutput);
    };
    let computed_reference;
    let reference = match reference_override {
        Some(reference) => reference,
        None => {
            let Some(reference) = atomic_root_c_reference(output, objectives) else {
                return refuse(AtomicCudaRowsRefusal::ReferenceNonFiniteOrInverted);
            };
            computed_reference = reference;
            &computed_reference
        }
    };
    let Some(alpha_state) = bootstrap.root_alpha_state.as_ref() else {
        return refuse(AtomicCudaRowsRefusal::MissingAlphaState);
    };
    let alpha_schedule = atomic_root_c_margin_alpha_schedule(
        &bootstrap.alpha_config,
        bootstrap.phase_cap_optimizer_updates,
    );

    AtomicCudaMarginStepRequest::new(
        graph,
        input,
        &output_name,
        &bootstrap.initial_node_bounds,
        alpha_state,
        spec_matrix,
        reference,
        thresholds,
        verify_upper_bound,
        !conjunctive,
        matches!(
            bootstrap.alpha_config.gradient_method,
            crate::bounds::GradientMethod::AnalyticChain
        ),
        matches!(
            bootstrap.alpha_config.optimizer,
            crate::bounds::Optimizer::Adam
        ),
        alpha_schedule.adam,
        multi_iterations,
        alpha_schedule.learning_rate_decay,
        alpha_schedule.bracket_base_learning_rate,
        deadline,
    )
    .run()
}

/// Stage-A dispatch token. Exactly one is built per root-C attempt and it is
/// consumed immediately by the flat Stage-A match, so the 144-byte variant
/// costs ~100 bytes of stack once and is never stored or collected.
// Not boxed: stable Rust cannot pattern-match through a `Box`, so boxing the
// margin payload would force the soundness-critical Stage-A match apart —
// including the two or-patterns that unify the `RefusedBeforeCommit` arms
// across both variants — for no measurable saving on a single short-lived
// local behind a CUDA dispatch.
#[allow(clippy::large_enum_variant)]
enum AtomicRootCStageAOutcome {
    Rows(AtomicCudaRowsOutcome),
    MarginStep(AtomicCudaMarginStepOutcome),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomicRootCRoute {
    Rows,
    Margin { multi_iterations: usize },
}

/// Typed iterations arm only the complete root-C margin transaction. They do
/// not change `root_alpha_cuda_rows_enabled`, so the separate identity-preloop
/// consumer stays byte-dark. With a typed zero, preserve the historical parent
/// rows gate and subordinate margin gate exactly.
fn select_atomic_root_c_route(
    typed_iterations: usize,
    legacy_rows_enabled: bool,
    legacy_margin_enabled: bool,
) -> Option<AtomicRootCRoute> {
    if typed_iterations > 0 {
        Some(AtomicRootCRoute::Margin {
            multi_iterations: typed_iterations,
        })
    } else if legacy_rows_enabled && legacy_margin_enabled {
        Some(AtomicRootCRoute::Margin {
            multi_iterations: 0,
        })
    } else if legacy_rows_enabled {
        Some(AtomicRootCRoute::Rows)
    } else {
        None
    }
}

fn atomic_root_c_route_accepts_compact_rows(
    route: AtomicRootCRoute,
    compact_rows_available: bool,
) -> bool {
    compact_rows_available
        && matches!(
            route,
            AtomicRootCRoute::Margin {
                multi_iterations
            } if multi_iterations > 0
        )
}

/// Attribute only the typed bounded exact-C route. Legacy environment-gated
/// rows/margin experiments deliberately remain outside this treatment counter.
fn observe_typed_exact_c_outcome(outcome: &AtomicRootCStageAOutcome) {
    let AtomicRootCStageAOutcome::MarginStep(outcome) = outcome else {
        crate::execution_telemetry::record_exact_c_attribution_conflict("unexpected_rows_route");
        return;
    };
    match outcome {
        AtomicCudaMarginStepOutcome::RefusedBeforeCommit { refusal } => {
            crate::execution_telemetry::record_exact_c_refused_before_commit(
                refusal.telemetry_reason(),
            );
        }
        AtomicCudaMarginStepOutcome::Committed(commit) => match commit {
            AtomicCudaMarginStepCommit::DeadlineExceeded => {
                crate::execution_telemetry::record_exact_c_committed(
                    None,
                    None,
                    "deadline_after_cleanup",
                );
            }
            AtomicCudaMarginStepCommit::Alpha0Retained { refusal, .. } => {
                crate::execution_telemetry::record_exact_c_committed(
                    None,
                    None,
                    refusal.telemetry_reason(),
                );
            }
            AtomicCudaMarginStepCommit::Alpha0Selected { .. } => {
                crate::execution_telemetry::record_exact_c_committed(None, None, "alpha0_selected");
            }
            AtomicCudaMarginStepCommit::Alpha1Selected { .. } => {
                crate::execution_telemetry::record_exact_c_committed(None, None, "alpha1_selected");
            }
            AtomicCudaMarginStepCommit::TopKAlpha0Selected { .. } => {
                crate::execution_telemetry::record_exact_c_committed(
                    None,
                    None,
                    "topk_alpha0_selected",
                );
            }
            AtomicCudaMarginStepCommit::TopKAlpha1Selected { .. } => {
                crate::execution_telemetry::record_exact_c_committed(
                    None,
                    None,
                    "topk_alpha1_selected",
                );
            }
            AtomicCudaMarginStepCommit::MultiAlpha0Selected {
                attempted_iterations,
                multiplicative_weights_requested,
                multiplicative_weights_plan_dispatched,
                multiplicative_weights_effective,
                completed_proposals,
                adaptive_plan_dispatches,
                gradient_plan_num_specs,
                gradient_row_count,
                stop_refusal,
                ..
            } => {
                let stop_reason = stop_refusal
                    .map(|refusal| refusal.telemetry_reason())
                    .unwrap_or("iteration_limit");
                crate::execution_telemetry::record_exact_c_multi_iteration_committed(
                    *attempted_iterations,
                    0,
                    *multiplicative_weights_requested,
                    *multiplicative_weights_plan_dispatched,
                    *multiplicative_weights_effective,
                    *completed_proposals,
                    *adaptive_plan_dispatches,
                    *gradient_plan_num_specs,
                    *gradient_row_count,
                    stop_reason,
                );
            }
            AtomicCudaMarginStepCommit::MultiAlphaSelected {
                attempted_iterations,
                accepted_iterations,
                multiplicative_weights_requested,
                multiplicative_weights_plan_dispatched,
                multiplicative_weights_effective,
                completed_proposals,
                adaptive_plan_dispatches,
                gradient_plan_num_specs,
                gradient_row_count,
                stop_refusal,
                ..
            } => {
                let stop_reason = stop_refusal
                    .map(|refusal| refusal.telemetry_reason())
                    .unwrap_or("iteration_limit");
                crate::execution_telemetry::record_exact_c_multi_iteration_committed(
                    *attempted_iterations,
                    *accepted_iterations,
                    *multiplicative_weights_requested,
                    *multiplicative_weights_plan_dispatched,
                    *multiplicative_weights_effective,
                    *completed_proposals,
                    *adaptive_plan_dispatches,
                    *gradient_plan_num_specs,
                    *gradient_row_count,
                    stop_reason,
                );
            }
        },
    }
}

#[cfg(test)]
mod atomic_root_c_margin_schedule_tests {
    use super::{
        atomic_root_c_margin_alpha_schedule, atomic_root_c_route_accepts_compact_rows,
        observe_typed_exact_c_outcome, select_atomic_root_c_route, AtomicRootCRoute,
        AtomicRootCStageAOutcome,
    };
    use crate::bounds::AlphaCrownConfig;
    use crate::network::{AtomicCudaMarginStepOutcome, AtomicCudaRowsRefusal};

    #[test]
    fn cifar_bracket_base_is_separate_from_the_dark_decayed_single_step() {
        let alpha_config = AlphaCrownConfig {
            learning_rate: 0.25,
            lr_decay: 0.98,
            iterations: 20,
            ..AlphaCrownConfig::default()
        };
        let schedule = atomic_root_c_margin_alpha_schedule(&alpha_config, None);
        let expected_next = 0.25 * 0.98_f32.powf(20.0);

        assert_eq!(
            schedule.adam.learning_rate.to_bits(),
            expected_next.to_bits(),
            "the existing single-step path must retain its next decayed LR"
        );
        assert_eq!(
            schedule.bracket_base_learning_rate.to_bits(),
            0.25_f32.to_bits(),
            "the bracket must receive the configured CIFAR base LR"
        );
        assert_ne!(
            schedule.adam.learning_rate.to_bits(),
            schedule.bracket_base_learning_rate.to_bits()
        );
        assert_eq!(schedule.learning_rate_decay.to_bits(), 0.98_f32.to_bits());
    }

    #[test]
    fn checkpoint_schedule_continues_from_actual_optimizer_update_count() {
        let alpha_config = AlphaCrownConfig {
            learning_rate: 0.25,
            lr_decay: 0.98,
            iterations: 20,
            ..AlphaCrownConfig::default()
        };
        let schedule = atomic_root_c_margin_alpha_schedule(&alpha_config, Some(3));
        assert_eq!(
            schedule.adam.learning_rate.to_bits(),
            (0.25 * 0.98_f32.powf(3.0)).to_bits()
        );
        assert_eq!(schedule.adam.t, 4);
    }

    #[test]
    fn typed_route_arms_only_root_margin_and_legacy_gates_remain_exact() {
        assert_eq!(select_atomic_root_c_route(0, false, false), None);
        assert_eq!(
            select_atomic_root_c_route(0, false, true),
            None,
            "the legacy margin child cannot arm its parent rows route"
        );
        assert_eq!(
            select_atomic_root_c_route(0, true, false),
            Some(AtomicRootCRoute::Rows)
        );
        assert_eq!(
            select_atomic_root_c_route(0, true, true),
            Some(AtomicRootCRoute::Margin {
                multi_iterations: 0
            })
        );
        assert_eq!(
            select_atomic_root_c_route(3, false, false),
            Some(AtomicRootCRoute::Margin {
                multi_iterations: 3
            }),
            "typed iterations must not depend on either legacy environment gate"
        );
        assert_eq!(
            select_atomic_root_c_route(8, true, true),
            Some(AtomicRootCRoute::Margin {
                multi_iterations: 8
            }),
            "the typed bounded route is authoritative once armed"
        );
    }

    #[test]
    fn only_typed_multi_iteration_route_accepts_compact_rows() {
        assert!(!atomic_root_c_route_accepts_compact_rows(
            AtomicRootCRoute::Rows,
            true,
        ));
        assert!(!atomic_root_c_route_accepts_compact_rows(
            AtomicRootCRoute::Margin {
                multi_iterations: 0,
            },
            true,
        ));
        assert!(!atomic_root_c_route_accepts_compact_rows(
            AtomicRootCRoute::Margin {
                multi_iterations: 4,
            },
            false,
        ));
        assert!(atomic_root_c_route_accepts_compact_rows(
            AtomicRootCRoute::Margin {
                multi_iterations: 4,
            },
            true,
        ));
    }

    #[test]
    fn typed_outcome_mapping_feeds_the_run_scoped_recorder() {
        let _test_lock = crate::execution_telemetry::TEST_LOCK
            .lock()
            .expect("telemetry test lock");
        let _run = crate::execution_telemetry::begin_run();
        crate::execution_telemetry::record_exact_c_selected(4, 3, 3, 0);
        observe_typed_exact_c_outcome(&AtomicRootCStageAOutcome::MarginStep(
            AtomicCudaMarginStepOutcome::RefusedBeforeCommit {
                refusal: AtomicCudaRowsRefusal::MissingDeadline,
            },
        ));

        let observed = crate::execution_telemetry::snapshot();
        assert_eq!(observed.exact_c.selections, 1);
        assert_eq!(observed.exact_c.outcomes_observed, 1);
        assert_eq!(observed.exact_c.refused_before_commit, 1);
        assert_eq!(observed.exact_c.stop_reasons["missing_deadline"], 1);
        assert!(!observed.exact_c.attribution_conflict);
    }
}

/// Select the alpha state eligible to seed root BaB after root-box tightening.
///
/// The ordinary warmup state predates optional intermediate tightening, so it
/// remains deliberately rejected when those boxes changed.  An atomic margin
/// alpha override is different: alpha1 was evaluated against the already-frozen
/// tightened boxes and can therefore cross that quality moat.  The established
/// `beta_iterations > 0` policy remains authoritative over both sources.
fn root_bab_alpha_warm_start(
    root_alpha: Option<&crate::bounds::GraphAlphaState>,
    root_boxes_tightened: bool,
    fresh_atomic_override: bool,
    warm_start_enabled: bool,
) -> Option<&crate::bounds::GraphAlphaState> {
    if !warm_start_enabled || (root_boxes_tightened && !fresh_atomic_override) {
        None
    } else {
        root_alpha
    }
}

#[cfg(test)]
mod atomic_root_alpha_transport_tests {
    use super::root_bab_alpha_warm_start;
    use crate::bounds::GraphAlphaState;

    #[test]
    fn tightened_boxes_admit_only_a_fresh_evaluated_override() {
        let alpha = GraphAlphaState::new();

        assert!(
            root_bab_alpha_warm_start(Some(&alpha), true, false, true).is_none(),
            "the pre-tightening warmup must remain stale"
        );
        let selected = root_bab_alpha_warm_start(Some(&alpha), true, true, true)
            .expect("the post-tightening evaluated alpha1 must remain paired for BaB");
        assert!(std::ptr::eq(selected, std::ptr::from_ref(&alpha)));
    }

    #[test]
    fn beta_iteration_policy_remains_the_final_warm_start_gate() {
        let alpha = GraphAlphaState::new();

        assert!(
            root_bab_alpha_warm_start(Some(&alpha), false, true, false).is_none(),
            "a fresh override must not bypass the established beta-iteration policy"
        );
        assert!(
            root_bab_alpha_warm_start(None, true, true, true).is_none(),
            "fresh provenance cannot manufacture a missing alpha state"
        );
    }
}

/// Build the multi-row spec objective that ranks root-warmup α iterates
/// (#root-alpha-margin).
///
/// Fails closed to `None` — the warmup then keeps its legacy last-iterate α — on
/// any of: the disjunctive mode, an empty or mismatched objective/threshold set,
/// or a non-finite coefficient.
///
/// CONJUNCTIVE POLARITY: `conjunctive == false` is the mode in which a domain is
/// verified only when ALL objectives hold (see `verify.rs:170-172`), which is
/// exactly when a hinge over every row is the right thing to maximize. Under
/// `conjunctive == true` (ANY objective suffices) the correct objective would be a
/// max over rows, not a hinge, so we decline rather than rank on the wrong scalar.
fn build_root_alpha_ascent(
    conjunctive: bool,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    verify_upper_bound: bool,
) -> Option<crate::bounds::AlphaSpecAscent> {
    if conjunctive || objectives.is_empty() || objectives.len() != thresholds.len() {
        return None;
    }
    let width = objectives.first()?.len();
    if width == 0
        || objectives
            .iter()
            .any(|o| o.len() != width || o.iter().any(|v| !v.is_finite()))
        || thresholds.iter().any(|t| !t.is_finite())
    {
        return None;
    }
    let rows = objectives
        .iter()
        .zip(thresholds)
        .map(
            |(objective, &threshold)| crate::bounds::AlphaSpecEarlyExit {
                objective: objective.clone(),
                threshold,
                verify_upper_bound,
            },
        )
        .collect();
    crate::bounds::AlphaSpecAscent::new(rows)
}

/// Build a root-spec compression plan, returning `None` to fail closed to the
/// existing full request. In particular, conjunctive semantics are never
/// compressed: their root stopping rule is "any row verified", unlike the
/// disjunctive all-rows rule this optimization targets.
fn build_root_spec_prune_plan(
    enabled: bool,
    conjunctive: bool,
    output: &BoundedTensor,
    full_spec_matrix: &ndarray::Array2<f32>,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
) -> Option<RootSpecPrunePlan> {
    if !enabled || conjunctive || objectives.is_empty() || objectives.len() != thresholds.len() {
        return None;
    }

    let output_dim = output.flatten().len();
    if full_spec_matrix.nrows() != objectives.len()
        || full_spec_matrix.ncols() != output_dim
        || objectives.iter().any(|objective| {
            objective.len() != output_dim || objective.iter().any(|value| !value.is_finite())
        })
        || thresholds.iter().any(|threshold| !threshold.is_finite())
    {
        return None;
    }

    let pre_bounds = BetaCrownVerifier::objective_bounds_multi(output, objectives).ok()?;
    if pre_bounds.len() != objectives.len() {
        return None;
    }

    // Strictly mirror the verifier's authority test (`lower > threshold`) while
    // requiring a finite, ordered enclosure. Malformed/degenerate arithmetic
    // must never create a shortcut.
    let active_indices: Vec<usize> = pre_bounds
        .iter()
        .zip(thresholds)
        .enumerate()
        .filter_map(|(idx, (&(lower, upper), &threshold))| {
            (!root_prebound_certifies(lower, upper, threshold)).then_some(idx)
        })
        .collect();

    let active_spec_matrix = if active_indices.is_empty() {
        None
    } else {
        let mut active = ndarray::Array2::zeros((active_indices.len(), output_dim));
        for (active_row, &full_row) in active_indices.iter().enumerate() {
            active
                .row_mut(active_row)
                .assign(&full_spec_matrix.row(full_row));
        }
        Some(active)
    };

    Some(RootSpecPrunePlan {
        bootstrap_output: output.clone(),
        pre_bounds,
        active_indices,
        active_spec_matrix,
    })
}

/// Restore active CROWN rows into the certified full pre-bound vector.
///
/// Both enclosures are independently sound, so take their intersection when it
/// is well formed.  This prevents compression from weakening an active row when
/// the optimized spec candidate happens to be looser than the bootstrap output
/// projection. A malformed or disjoint compact candidate rejects the whole
/// compact result so its cache cannot gain downstream authority. Before an
/// atomic boundary the caller may retry the full legacy request; afterward it
/// retains the bootstrap vector. A finite compact row may replace a malformed
/// bootstrap row because the former is an independent certified enclosure.
fn merge_root_spec_pruned_bounds(
    plan: &RootSpecPrunePlan,
    active_bounds: Vec<(f32, f32)>,
) -> Option<Vec<(f32, f32)>> {
    if active_bounds.len() != plan.active_indices.len() {
        return None;
    }
    let mut seen = vec![false; plan.pre_bounds.len()];
    let mut merged = plan.pre_bounds.clone();
    for (&full_idx, (active_lower, active_upper)) in plan.active_indices.iter().zip(active_bounds) {
        if full_idx >= merged.len() || seen[full_idx] {
            return None;
        }
        seen[full_idx] = true;
        let pre = merged[full_idx];
        let active = (active_lower, active_upper);
        match (
            root_interval_is_finite_ordered(pre.0, pre.1),
            root_interval_is_finite_ordered(active.0, active.1),
        ) {
            (true, true) => {
                let intersection = (pre.0.max(active.0), pre.1.min(active.1));
                if root_interval_is_finite_ordered(intersection.0, intersection.1) {
                    merged[full_idx] = intersection;
                } else {
                    // The compact run's bounds and bootstrap enclosure must
                    // describe the same reachable values. Reject the entire
                    // compact result (including its cache) on disagreement.
                    return None;
                }
            }
            // A malformed compact result also invalidates any captured linear
            // cache or pending alpha as downstream authority.
            (true, false) => return None,
            (false, true) => {
                // The row was deliberately kept active because its bootstrap
                // interval could not certify anything. A fresh finite CROWN
                // enclosure is independently sound and can replace it.
                merged[full_idx] = active;
            }
            (false, false) => return None,
        }
    }
    merged
        .iter()
        .all(|&(lower, upper)| root_interval_is_finite_ordered(lower, upper))
        .then_some(merged)
}

fn f32_rows_bit_identical(left: impl IntoIterator<Item = f32>, right: &[f32]) -> bool {
    let mut left = left.into_iter();
    for &right_value in right {
        let Some(left_value) = left.next() else {
            return false;
        };
        if left_value.to_bits() != right_value.to_bits() {
            return false;
        }
    }
    left.next().is_none()
}

/// Bind every compact exact-C input to the same strict source-row map.
///
/// This is deliberately stricter than ordinary spec compression: exact-C may
/// return a selected alpha state whose binding-row provenance must remain tied
/// to the original objective.  Any duplicate, reordering, shape drift, bitwise
/// C/objective disagreement, or unusable reference interval declines the typed
/// composition before backend commitment.
fn build_atomic_root_c_compact_rows(
    plan: &RootSpecPrunePlan,
    full_spec_matrix: &ndarray::Array2<f32>,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
) -> Option<AtomicRootCCompactRows> {
    let source_rows = objectives.len();
    let active_rows = plan.active_indices.len();
    let active_spec_matrix = plan.active_spec_matrix.as_ref()?;
    if source_rows == 0
        || active_rows == 0
        || thresholds.len() != source_rows
        || plan.pre_bounds.len() != source_rows
        || full_spec_matrix.nrows() != source_rows
        || active_spec_matrix.nrows() != active_rows
        || active_spec_matrix.ncols() != full_spec_matrix.ncols()
        || objectives
            .iter()
            .enumerate()
            .any(|(source_row, objective)| {
                objective.len() != full_spec_matrix.ncols()
                    || objective.iter().any(|value| !value.is_finite())
                    || !f32_rows_bit_identical(
                        full_spec_matrix.row(source_row).iter().copied(),
                        objective,
                    )
            })
    {
        return None;
    }

    // Re-derive the unresolved set instead of trusting a partial/reordered map
    // merely because its shape happens to match the compact C matrix.
    let mut expected_compact_row = 0usize;
    for (source_row, (&(lower, upper), &threshold)) in
        plan.pre_bounds.iter().zip(thresholds).enumerate()
    {
        if !root_prebound_certifies(lower, upper, threshold) {
            if plan.active_indices.get(expected_compact_row) != Some(&source_row) {
                return None;
            }
            expected_compact_row += 1;
        }
    }
    if expected_compact_row != active_rows {
        return None;
    }

    let mut compact_objectives = Vec::with_capacity(active_rows);
    let mut compact_thresholds = Vec::with_capacity(active_rows);
    let mut reference_lower = Vec::with_capacity(active_rows);
    let mut reference_upper = Vec::with_capacity(active_rows);
    let mut previous_source = None;
    for (compact_row, &source_row) in plan.active_indices.iter().enumerate() {
        if source_row >= source_rows
            || previous_source.is_some_and(|previous| source_row <= previous)
        {
            return None;
        }
        previous_source = Some(source_row);

        let objective = objectives.get(source_row)?;
        if !f32_rows_bit_identical(
            active_spec_matrix.row(compact_row).iter().copied(),
            objective,
        ) {
            return None;
        }
        let threshold = *thresholds.get(source_row)?;
        let (lower, upper) = *plan.pre_bounds.get(source_row)?;
        if !threshold.is_finite() || !root_interval_is_finite_ordered(lower, upper) {
            return None;
        }

        compact_objectives.push(objective.clone());
        compact_thresholds.push(threshold);
        reference_lower.push(lower);
        reference_upper.push(upper);
    }

    let reference = BoundedTensor::new(
        ndarray::Array1::from_vec(reference_lower).into_dyn(),
        ndarray::Array1::from_vec(reference_upper).into_dyn(),
    )
    .ok()?;
    Some(AtomicRootCCompactRows {
        source_indices: plan.active_indices.clone(),
        objectives: compact_objectives,
        thresholds: compact_thresholds,
        spec_matrix: active_spec_matrix.clone(),
        reference,
    })
}

fn map_compact_atomic_root_c_row(source_indices: &[usize], compact_row: usize) -> Option<usize> {
    source_indices.get(compact_row).copied()
}

fn map_compact_atomic_root_c_rows(
    source_indices: &[usize],
    compact_rows: &[usize],
) -> Option<Vec<usize>> {
    compact_rows
        .iter()
        .map(|&row| map_compact_atomic_root_c_row(source_indices, row))
        .collect()
}

/// A pre-commit compact failure may use the historical full-spec fallback. An
/// atomic compact commit may not: its complete bootstrap reference remains the
/// only fallback authority after the backend boundary has been crossed.
fn retry_full_spec_after_compact_merge_failure(atomic_compact_committed: bool) -> bool {
    !atomic_compact_committed
}

fn publish_pending_atomic_root_alpha(
    pending: Option<crate::bounds::GraphAlphaState>,
    atomic_compact_committed: bool,
    compact_reconstruction_succeeded: bool,
) -> Option<crate::bounds::GraphAlphaState> {
    if atomic_compact_committed && !compact_reconstruction_succeeded {
        None
    } else {
        pending
    }
}

fn compact_atomic_root_c_reconstruction_succeeded(
    atomic_compact_committed: bool,
    merged_bounds_available: bool,
    binding_map_valid: bool,
) -> bool {
    !atomic_compact_committed || (merged_bounds_available && binding_map_valid)
}

struct RootSpecPrunePublication {
    /// `None` requests the historical full-spec retry. A committed compact
    /// failure instead carries the complete bootstrap pre-bound vector.
    bounds: Option<Vec<(f32, f32)>>,
    alpha: Option<crate::bounds::GraphAlphaState>,
    retry_full_spec: bool,
    bounds_reconstruction_succeeded: bool,
}

/// Resolve the entire compact publication transaction in one place. This
/// prevents a reconstruction or binding failure from publishing a selected
/// alpha, and prevents a post-commit failure from opening a second backend
/// transaction. The independently certified bootstrap vector is authoritative
/// in that case.
fn finalize_root_spec_prune_publication(
    plan: &RootSpecPrunePlan,
    active_bounds: Vec<(f32, f32)>,
    pending_alpha: Option<crate::bounds::GraphAlphaState>,
    atomic_compact_committed: bool,
    binding_map_valid: bool,
) -> RootSpecPrunePublication {
    let merged = merge_root_spec_pruned_bounds(plan, active_bounds);
    let bounds_reconstruction_succeeded = merged.is_some();
    let compact_reconstruction_succeeded = compact_atomic_root_c_reconstruction_succeeded(
        atomic_compact_committed,
        bounds_reconstruction_succeeded,
        binding_map_valid,
    );
    let alpha = publish_pending_atomic_root_alpha(
        pending_alpha,
        atomic_compact_committed,
        compact_reconstruction_succeeded,
    );

    if atomic_compact_committed && !compact_reconstruction_succeeded {
        RootSpecPrunePublication {
            bounds: Some(plan.pre_bounds.clone()),
            alpha,
            retry_full_spec: false,
            bounds_reconstruction_succeeded,
        }
    } else if let Some(bounds) = merged {
        RootSpecPrunePublication {
            bounds: Some(bounds),
            alpha,
            retry_full_spec: false,
            bounds_reconstruction_succeeded,
        }
    } else {
        RootSpecPrunePublication {
            bounds: None,
            alpha: None,
            retry_full_spec: retry_full_spec_after_compact_merge_failure(atomic_compact_committed),
            bounds_reconstruction_succeeded,
        }
    }
}

pub(super) fn validate_multi_objective_inputs(
    objectives: &[Vec<f32>],
    thresholds: &[f32],
) -> Result<()> {
    if objectives.is_empty() {
        return Err(ny_core::NyError::InvalidSpec(
            "empty objectives in multi-objective verification — nothing to verify".to_string(),
        ));
    }
    if objectives.len() != thresholds.len() {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "objectives/thresholds length mismatch: {} objectives vs {} thresholds (#3383)",
            objectives.len(),
            thresholds.len()
        )));
    }
    Ok(())
}

/// The retry must have materially MORE room than the window that just failed.
///
/// A flat 30 s floor was wrong and measured harmful. At the official 100 s
/// budget the phase cap expired with `global_left = 36.5 s`, which cleared a
/// 30 s floor but was nowhere near enough to finish an uncapped root: the retry
/// spent 37.6 s more, still did not admit the forward-linear build, and starved
/// the downstream margin-row lane from 48.2 s to 10.5 s. Verdict unchanged, so
/// it was pure loss.
///
/// Requiring `global_left >= 3x the expired cap` separates the two measured
/// regimes cleanly: 185.8 s available at a 330 s instance (>= 3x40 = 120, fires,
/// recovers 186 s) versus 36.5 s at 100 s (declines).
const ROOT_CAP_RETRY_HEADROOM_MULTIPLE: u32 = 3;

/// Fraction of the remaining global budget the retry may consume, so it always
/// leaves a BaB floor rather than converting the whole instance into one long
/// root pass.
const ROOT_CAP_RETRY_MAX_FRAC: f64 = 0.6;

/// #root-cap-degrade gate. Exact `"1"`, read once. Selects which certified
/// bound collector runs, never the arithmetic of a bound.
fn root_cap_degrade_enabled() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("NY_ROOT_CAP_DEGRADE").is_ok_and(|v| v == "1"))
}

/// #root-cap-retry gate. Exact `"1"`, read once -- it selects scheduling, never
/// a bound.
fn root_cap_retry_enabled() -> bool {
    static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var("NY_ROOT_CAP_RETRY").is_ok_and(|v| v == "1"))
}

/// Resolve the root-only checkpoint policy from its typed preset and exact env
/// override. An absent value inherits the typed setting, literal `1` enables,
/// and every other present byte string disables (the strict fail-closed preset
/// gate contract).
fn root_alpha_phase_checkpoint_from_raw(configured: bool, raw: Option<&std::ffi::OsStr>) -> bool {
    raw.map_or(configured, |value| value == std::ffi::OsStr::new("1"))
}

fn root_alpha_phase_checkpoint_enabled(config: &BetaCrownConfig) -> bool {
    root_alpha_phase_checkpoint_from_raw(
        config.root_alpha_phase_checkpoint,
        std::env::var_os("NY_ROOT_ALPHA_PHASE_CHECKPOINT").as_deref(),
    )
}

/// Rebase root-objective work after a consumed warmup deadline while never
/// extending beyond the verifier's global wall-clock authority.
fn resolve_root_objective_deadline(
    warmup_deadline: Option<std::time::Instant>,
    now: std::time::Instant,
    grace_slice: std::time::Duration,
    global_deadline: Option<std::time::Instant>,
    extend_live_deadline: bool,
) -> Option<std::time::Instant> {
    let grace = now.checked_add(grace_slice).unwrap_or(now);
    let cap_to_global = |candidate: std::time::Instant| {
        global_deadline.map_or(candidate, |global| candidate.min(global))
    };
    match warmup_deadline {
        Some(deadline) if deadline > now => {
            if extend_live_deadline {
                Some(cap_to_global(deadline.max(grace)))
            } else {
                Some(cap_to_global(deadline))
            }
        }
        Some(expired) => {
            let capped_grace = cap_to_global(grace);
            Some(if capped_grace > now {
                capped_grace
            } else {
                cap_to_global(expired)
            })
        }
        None => None,
    }
}

#[cfg(test)]
mod root_alpha_phase_checkpoint_tests {
    use super::{resolve_root_objective_deadline, root_alpha_phase_checkpoint_from_raw};
    use std::time::{Duration, Instant};

    #[test]
    fn exact_env_override_wins_in_both_directions() {
        assert!(!root_alpha_phase_checkpoint_from_raw(false, None));
        assert!(root_alpha_phase_checkpoint_from_raw(true, None));
        assert!(root_alpha_phase_checkpoint_from_raw(
            false,
            Some(std::ffi::OsStr::new("1"))
        ));
        assert!(!root_alpha_phase_checkpoint_from_raw(
            true,
            Some(std::ffi::OsStr::new("0"))
        ));
    }

    #[test]
    fn every_other_present_value_disables() {
        for malformed in ["", "01", " 1", "1 ", "true", "yes"] {
            assert!(
                !root_alpha_phase_checkpoint_from_raw(false, Some(std::ffi::OsStr::new(malformed))),
                "malformed gate value {malformed:?} must not arm a dark preset"
            );
            assert!(
                !root_alpha_phase_checkpoint_from_raw(true, Some(std::ffi::OsStr::new(malformed))),
                "malformed gate value {malformed:?} must fail closed"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_gate_fails_closed() {
        use std::os::unix::ffi::OsStringExt;

        let raw = std::ffi::OsString::from_vec(vec![0xff]);
        assert!(!root_alpha_phase_checkpoint_from_raw(
            false,
            Some(raw.as_os_str())
        ));
        assert!(!root_alpha_phase_checkpoint_from_raw(
            true,
            Some(raw.as_os_str())
        ));
    }

    #[test]
    fn root_objective_grace_rebases_expired_checkpoint_and_respects_global_wall() {
        let now = Instant::now();
        let expired = now.checked_sub(Duration::from_secs(1)).expect("expired");
        let live_warmup = now.checked_add(Duration::from_secs(1)).expect("live");
        let global = now.checked_add(Duration::from_secs(2)).expect("global");
        let beyond_global = now
            .checked_add(Duration::from_secs(4))
            .expect("beyond global");
        let grace = Duration::from_secs(3);

        assert_eq!(
            resolve_root_objective_deadline(Some(expired), now, grace, Some(global), false),
            Some(global),
            "expired local checkpoint deadline must rebase under live outer authority"
        );
        assert_eq!(
            resolve_root_objective_deadline(Some(live_warmup), now, grace, Some(global), false),
            Some(live_warmup),
            "ordinary live warmup keeps its historical deadline"
        );
        assert_eq!(
            resolve_root_objective_deadline(Some(live_warmup), now, grace, Some(global), true),
            Some(global),
            "alpha rebuild may extend a live warmup only to the capped grace"
        );
        assert_eq!(
            resolve_root_objective_deadline(Some(beyond_global), now, grace, Some(global), false),
            Some(global),
            "even an inherited live deadline must not exceed outer authority"
        );
        assert_eq!(
            resolve_root_objective_deadline(Some(beyond_global), now, grace, Some(global), true),
            Some(global),
            "extended root work must remain capped by outer authority"
        );
        assert_eq!(
            resolve_root_objective_deadline(None, now, grace, Some(global), true),
            None,
            "historically unbounded callers remain unbounded"
        );
        assert_eq!(
            resolve_root_objective_deadline(Some(expired), now, grace, Some(expired), false),
            Some(expired),
            "an exhausted outer authority must not mint new time"
        );
    }
}

/// Resolve comprehensive ownership exactly once. `Some` means the separate
/// comprehensive lever owned the slot, including a completed or declined
/// zero-result; the legacy closure is reachable only when that lever was
/// absent at orchestration time.
fn comprehensive_gpu_or_legacy_wide(
    comprehensive: Option<usize>,
    legacy: impl FnOnce() -> usize,
) -> usize {
    comprehensive.unwrap_or_else(legacy)
}

/// Resolve the phase-resident owner at the existing comprehensive slot.
///
/// A resident policy claims the slot without executing there: `Some(0)` keeps
/// the established wide-route ownership sentinel while the typed policy token
/// is carried to the later dense-head site. With no resident policy, the
/// legacy closure is evaluated exactly once and returned unchanged.
fn phase_resident_or_comprehensive(
    resident: Option<RootPhaseResidentCrownPolicy>,
    legacy: impl FnOnce() -> Option<usize>,
) -> (Option<RootPhaseResidentCrownPolicy>, Option<usize>) {
    match resident {
        Some(policy) => (Some(policy), Some(0)),
        None => (None, legacy()),
    }
}

/// Whether any root intermediate-bound producer changed the frozen boxes.
///
/// Keep this as the single source of truth for every downstream stale-alpha
/// decision. In particular, comprehensive and phase-resident GPU routes mutate
/// `bootstrap.initial_node_bounds` before the root objective is evaluated; if
/// either realized target count is omitted here, later consumers can
/// incorrectly reuse alpha state built for the pre-treatment boxes.
fn root_intermediate_tightening_changed(
    dense_phase: super::root_phases::PhaseOutput,
    comprehensive_gpu_targets: Option<usize>,
    wide_demanded_targets: usize,
    sparse_targets: usize,
    joint_alpha_targets: usize,
) -> bool {
    dense_phase.bounds_changed()
        || comprehensive_gpu_targets.is_some_and(|targets| targets > 0)
        || wide_demanded_targets > 0
        || sparse_targets > 0
        || joint_alpha_targets > 0
}

pub(super) fn evaluate_root<'property>(
    request: MultiObjectiveRootRequest<'_, 'property>,
    lifecycle: &mut GraphBabLifecycle,
) -> Result<MultiObjectiveRootEvaluation<'property>> {
    let MultiObjectiveRootRequest {
        verifier,
        graph,
        input,
        property,
        engine,
        conjunctive,
        deadline,
    } = request;
    let outcome = {
        let (objectives, thresholds) = property.views();
        evaluate_root_borrowed(
            BorrowedMultiObjectiveRootRequest {
                verifier,
                graph,
                input,
                objectives,
                thresholds,
                engine,
                conjunctive,
                deadline,
            },
            lifecycle,
        )?
    };
    Ok(MultiObjectiveRootEvaluation { outcome, property })
}

fn evaluate_root_borrowed(
    request: BorrowedMultiObjectiveRootRequest<'_, '_>,
    lifecycle: &mut GraphBabLifecycle,
) -> Result<MultiObjectiveRootOutcome> {
    let BorrowedMultiObjectiveRootRequest {
        verifier,
        graph,
        input,
        objectives,
        thresholds,
        engine,
        conjunctive,
        deadline,
    } = request;
    let critical_gpu_alpha_enabled = root_critical_gpu_alpha_enabled();
    // Warmup cap (#2206 Packet C, #4095): initial bounds get at most
    // `initial_bounds_fraction` of the BaB timeout. Mirrors core.rs pattern.
    //
    // When a wall-clock deadline is provided (#4321), derive the effective
    // timeout from remaining time instead of the configured timeout.
    let pgd_frac = verifier
        .config
        .phase_budget
        .post_bab_pgd_fraction
        .clamp(0.0, 0.5);
    // #cora-double-reserve: Some(deadline) already carries the ledger's one-time
    // post_bab_pgd_fraction reservation — do not scale it again (see verify.rs).
    let bab_timeout = match deadline {
        Some(dl) => dl.saturating_duration_since(lifecycle.start_time),
        None => verifier.config.timeout.mul_f32(1.0 - pgd_frac),
    };
    // The mandatory foundational node-bounds sweep must reach every node, so it
    // gets the full global deadline (not the warmup fraction) — capping it choked
    // conv-heavy DAGs (yolo, tinyimagenet) into "deadline exceeded before node
    // 'Conv_0'" with most of the budget unused (#4321).
    let bab_deadline = lifecycle.deadline(bab_timeout);
    // The public convenience entry point supplies no explicit deadline, but
    // the configured timeout is still authoritative. Thread that derived
    // boundary through every optional root phase too; otherwise those phases
    // see `None` as "unbounded" and may grant themselves fresh grace slices
    // after the configured budget has already expired. A caller-provided
    // ledger deadline remains exact because `bab_deadline` is derived from it.
    let deadline = Some(deadline.map_or(bab_deadline, |caller| caller.min(bab_deadline)));
    let initial_deadline = Some(bab_deadline);
    // #w4-root-alpha-opt WARMUP CAP: the alpha warmup otherwise runs to the
    // full BaB budget (the initial_bounds_fraction knob never capped this
    // path — W4-2 finding), leaving the root pass a grace slice smaller than
    // one measured forward-map rebuild (~22s grace vs ~25s rebuild at 95s,
    // measured), so the root alpha OPTIMIZER — the root-relaxation lever —
    // could never fire. When the lever is armed and the CLI attack-phase
    // warmer measured the fixed-map cost, reserve optimizer+rebuild budget
    // out of the warmup. The worst stragglers are BETA-INSENSITIVE (W4-6),
    // so trading warmup iterations for root tightness is the measured-win
    // direction. Applies only where the fixed forward map is warm (image
    // conv DAGs); everything else keeps the status quo. Sound: deadlines
    // only schedule work.
    let initial_deadline = {
        let alpha_lever_armed =
            graph.has_conv_layers() && graph.forward_linear_spec_alpha_enabled();
        let reserved_deadline = if alpha_lever_armed {
            graph
                .forward_linear_fixed_pass_cost(input)
                .zip(deadline)
                .and_then(|(cost, global)| {
                    // Rebuild (~= fixed cost) with margin + optimizer sweeps
                    // + the root spec pass's own candidates.
                    let reserve = cost.mul_f64(1.3) + std::time::Duration::from_secs(10);
                    global.checked_sub(reserve)
                })
        } else {
            None
        };
        match (initial_deadline, reserved_deadline) {
            (Some(base), Some(reserved)) => {
                // Warmup floor: keep enough for the reference-bounds pass and
                // a few alpha iterations (the GPU root candidate still wants
                // a sane alpha state).
                let floor = std::time::Instant::now() + std::time::Duration::from_secs(10);
                Some(base.min(reserved.max(floor)))
            }
            (base, _) => base,
        }
    };
    // #scored-budget-discriminator (dark, print-only): expose the ACTUAL
    // multi-objective root allocation. In particular, this records the
    // remaining-time-derived warmup window selected above beside the preset's
    // `initial_bounds_fraction`; the latter is not the controlling cap on this
    // path today. A matched 100s/200s trace can therefore distinguish
    // allocation/alpha-state effects from actual per-domain GPU throughput.
    // No field below feeds a schedule, bound, or verdict.
    if crate::phase_telemetry::phase_telemetry_enabled() {
        let marker_now = std::time::Instant::now();
        let global_remaining = deadline
            .map(|value| value.saturating_duration_since(marker_now).as_secs_f64())
            .unwrap_or(f64::INFINITY);
        let warmup_remaining = initial_deadline
            .map(|value| value.saturating_duration_since(marker_now).as_secs_f64())
            .unwrap_or(f64::INFINITY);
        crate::phase_telemetry::phase_marker(&format!(
            "multiobj-budget objectives={} effective-bab={:.3}s global-remaining={global_remaining:.3}s \
             selected-warmup-remaining={warmup_remaining:.3}s initial-bounds-fraction={:.3} \
             alpha-iters={} use-alpha={}",
            objectives.len(),
            bab_timeout.as_secs_f64(),
            verifier.config.phase_budget.initial_bounds_fraction,
            verifier.config.alpha_config.iterations,
            verifier.config.use_alpha_crown,
        ));
    }
    // Certified sparse-input double-double zonotope (#dd-zonotope, default-ON
    // for the fail-closed detector; `NY_DD_ZONOTOPE=0` is the kill switch).
    // Runs BEFORE the bootstrap because on
    // the category it targets (vggnet16_2022) the existing root pass consumes
    // the entire budget — an intersect placed after `compute_root_objective_bounds`
    // would never be reached. The explicit kill switch short-circuits before any
    // allocation. See `dd_zono_root` for the full
    // soundness contract: conjunctive detector, self-policing precision gate,
    // fail-closed refusals, and INTERSECT-never-replace publication.
    let dd_zono = run_dd_zono_root(graph, input, objectives, deadline, &verifier.config);
    if let Some(result) = dd_zono.as_ref() {
        // Certified-at-root fast exit. Requiring EVERY objective to clear its
        // threshold is sufficient for both the conjunctive (any) and the
        // disjunctive (all) verdict rules, so no rule-specific reasoning is
        // duplicated here.
        //
        // The three-way length equality is asserted rather than assumed. A
        // `zip` between a short `thresholds` and a long margin list would
        // silently check only the common prefix and then report `all` — i.e. it
        // would publish `Verified` for objectives it never examined. The
        // caller already rejects a mismatch (#3383) and `evaluate_objectives`
        // emits exactly one entry per objective, so this can only fire if one
        // of those invariants is later broken; it is a verdict site, so it
        // fails closed instead of trusting them.
        let safety = crate::dd_zonotope::DdZonoConfig::from_env().safety_factor;
        let lengths_agree = result.margin.lower.len() == thresholds.len()
            && result.margin.lower.len() == objectives.len();
        // Narrow OUTWARD (toward -inf) on the f64 -> f32 cast: a nearest-mode
        // cast could round a certified lower bound UP across the threshold.
        let all_verified = lengths_agree
            && thresholds.iter().enumerate().all(|(i, &t)| {
                let lo = result.margin.lower_with_safety(i, safety);
                lo.is_finite() && next_down_f32(lo as f32) > t
            });
        if all_verified {
            if let Some(output) = result.output.clone() {
                info!(
                    "#dd-zonotope: all {} objective(s) certified at the root by the \
                     double-double zonotope ({} generators, {:.1}s) — property safe",
                    objectives.len(),
                    result.margin.n_generators,
                    result.margin.wall.as_secs_f32()
                );
                lifecycle.domains_explored = 1;
                lifecycle.domains_verified = 1;
                return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
                    lifecycle.build_result_with_bounds(BabVerificationStatus::Verified, output),
                )));
            }
        }
    }

    // Root/output-bound passes now check the wall-clock deadline between nodes
    // (#4321). When one of them aborts because the deadline passed mid-phase, that
    // surfaces as DeadlineExceeded; convert it here into a graceful Timeout verdict
    // so the CLI emits a valid JSON result instead of being killed externally.
    // Sound: a Timeout never claims Verified.
    // #root-alpha-margin WIRING FIX: on this path the ONLY α warmup runs INSIDE
    // `compute_graph_bab_bootstrap` (measured ordering in one run's stderr:
    // `graph-bab-bootstrap start` → `gate ARMED, spec objective ABSENT` →
    // `dag-alpha-warmup loop-enter` → `loop-exit` → `graph-bab-bootstrap end`;
    // exactly one loop-enter). Attaching the spec rows to the RETURNED
    // bootstrap's `alpha_config` — as the block below does — therefore lands
    // AFTER the ascent it is meant to steer has already finished, and the gate
    // reports `spec objective ABSENT — INERT` on every cifar100 row.
    //
    // So build the ascent first and hand it to the config the bootstrap
    // actually consumes. `Cow::Borrowed` keeps the disarmed path allocation-free
    // and byte-identical; only an armed gate pays for the clone.
    // The attribution producer also needs the exact property rows even when
    // root-alpha margin selection itself is dark.  Supplying metadata in the
    // config does not arm that optimizer lane; it merely lets the separately
    // gated, <=3-row attribution fold seed the exact property matrix. Thus a
    // single `NY_ATTR_BRANCH=1` is sufficient to produce the prior it asks the
    // KFSB consumer to use, without implicitly changing alpha selection.
    let root_alpha_spec = (crate::network::root_alpha_margin_enabled_with(
        verifier.config.alpha_config.root_alpha_margin,
    ) || crate::network::gap_attribution::root_gap_probe_enabled())
    .then(|| {
        build_root_alpha_ascent(
            conjunctive,
            objectives,
            thresholds,
            verifier.config.verify_upper_bound,
        )
    })
    .flatten();
    let bootstrap_config = match root_alpha_spec.as_ref() {
        None => std::borrow::Cow::Borrowed(&verifier.config),
        Some(spec) => {
            let mut config = verifier.config.clone();
            config.alpha_config.spec_ascent = Some(spec.clone());
            std::borrow::Cow::Owned(config)
        }
    };
    // #margin-subset-seed on the MULTI-OBJECTIVE path.
    //
    // The single-objective relu-split loop publishes the spec-referenced OUTPUT
    // indices around its bootstrap (`relu_split/bab_loop.rs:735`), which is what
    // lets the CROWN-IBP collector seed only those rows at the OUTPUT node. The
    // conjunctive multi-objective path never did, so `margin_subset_indices`
    // returned `None` and subset seeding was disengaged on every instance that
    // takes this route — including yolo_2023, whose five objectives touch five
    // of 21,125 outputs.
    //
    // The union over all objectives' nonzero coefficient positions is the right
    // set here: the collection is shared by every objective, so a row any
    // objective reads must be tightened. `publish` sorts and deduplicates.
    //
    // Sound: this only selects which rows get the TIGHTER treatment. Rows
    // outside the set keep the node's existing IBP/forward bounds, which are
    // valid enclosures (see `output_margin_seed`'s scatter contract), so a
    // too-small publication costs tightness and never validity. An empty union
    // publishes nothing and leaves the full-width path byte-identical.
    let _margin_seed_guard = {
        let mut referenced: Vec<usize> = Vec::new();
        for objective in objectives {
            referenced.extend(
                objective
                    .iter()
                    .enumerate()
                    .filter(|(_, &coeff)| coeff != 0.0)
                    .map(|(idx, _)| idx),
            );
        }
        referenced.sort_unstable();
        referenced.dedup();
        tracing::info!(
            "#margin-subset-seed: multi-objective root publishing {} spec-referenced OUTPUT \
             indices from {} objectives",
            referenced.len(),
            objectives.len(),
        );
        crate::output_margin_seed::MarginOutputSeedGuard::publish(referenced)
    };

    let checkpoint_enabled = root_alpha_phase_checkpoint_enabled(&bootstrap_config);
    let bootstrap_result = if checkpoint_enabled {
        compute_graph_bab_bootstrap_with_phase_cap_checkpoint(
            graph,
            input,
            &bootstrap_config,
            engine,
            initial_deadline,
            deadline,
        )
    } else {
        compute_graph_bab_bootstrap(graph, input, &bootstrap_config, engine, initial_deadline)
    };
    let mut bootstrap = match bootstrap_result {
        Ok(bootstrap) => bootstrap,
        Err(ny_core::NyError::DeadlineExceeded(_)) => {
            // #root-cap-retry (DARK, NY_ROOT_CAP_RETRY=1).
            //
            // The bootstrap's deadline is the ROOT-ALPHA PHASE CAP
            // (`root_alpha_cap_secs`, a fixed 40 s on cifar100_2024), not the
            // global BaB deadline. Treating its expiry as whole-run exhaustion
            // discards every second of live budget: measured on
            // prop_idx_7500 at a 330 s instance, this returns Timeout with
            // domains_explored=0 and 209.9 s still on the ledger, which is why
            // the branch selector was never reached at all.
            //
            // The single-objective sibling (relu_split/bab_loop.rs:168-179)
            // distinguishes the two deadlines, but only to relabel Timeout ->
            // Unknown; it also returns. So no path re-enters BaB after a
            // phase-cap expiry.
            //
            // When the GLOBAL deadline is still live, retry the bootstrap once
            // without the phase cap. Sound: deadlines only schedule work, and
            // the retry is bounded by the global deadline it is handed.
            let now = std::time::Instant::now();
            let global_left = bab_deadline.saturating_duration_since(now);
            let expired_cap = bootstrap_config
                .root_alpha_cap_secs
                .filter(|c| c.is_finite() && *c > 0.0)
                .map_or(
                    std::time::Duration::ZERO,
                    std::time::Duration::from_secs_f64,
                );
            let required = expired_cap * ROOT_CAP_RETRY_HEADROOM_MULTIPLE;
            // Leave a BaB floor: the retry gets at most a fraction of what is
            // left, never the whole remainder.
            let retry_deadline = now + global_left.mul_f64(ROOT_CAP_RETRY_MAX_FRAC);
            if root_cap_retry_enabled() && global_left >= required && !required.is_zero() {
                let mut uncapped = bootstrap_config.as_ref().clone();
                uncapped.root_alpha_cap_secs = None;
                tracing::info!(
                    global_left_s = global_left.as_secs_f64(),
                    required_s = required.as_secs_f64(),
                    retry_window_s = (retry_deadline - now).as_secs_f64(),
                    "#root-cap-retry: root-alpha phase cap expired with global budget live; \
                     retrying the bootstrap uncapped"
                );
                match compute_graph_bab_bootstrap(
                    graph,
                    input,
                    &uncapped,
                    engine,
                    Some(retry_deadline),
                ) {
                    Ok(bootstrap) => bootstrap,
                    Err(ny_core::NyError::DeadlineExceeded(_)) => {
                        return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
                            lifecycle.timeout_result(),
                        )));
                    }
                    Err(e) => return Err(e),
                }
            } else if matches!(
                ny_core::phase_yield::classify_expiry(now, Some(bab_deadline), Some(bab_deadline)),
                ny_core::phase_yield::Expiry::PhaseOnly
            ) || root_cap_degrade_enabled()
            {
                // #root-cap-degrade: the CHEAP path to the same place.
                //
                // The alpha collection deliberately propagates DeadlineExceeded
                // (shared/init.rs:261-264, "Do NOT swallow it into an IBP
                // fallback here") because the single-objective and GPU BaB
                // entries translate it into a warmup-cap Unknown. This lane does
                // not translate it -- it discards the instance.
                //
                // But every sibling arm of that same chain already degrades to
                // certified IBP on a deadline (init.rs:280-284, :289-293), and
                // this preset runs with fix_interm_bounds=true, so disabling
                // alpha lands on exactly that IBP branch. IBP over this graph is
                // 0.182 GMAC -- 1/3072 of the forward-linear build.
                //
                // Why it matters far more than it looks: reaching a bootstrap AT
                // ALL is what reaches the dense-head CROWN tightener further down
                // this function. That pass is measured at 0.5 s to close 97.82%
                // of the ny-vs-abc root head-width gap, halving unstable ReLUs
                // 44 -> 22 and taking the root from 94/99 to 98/99 rows
                // (docs/CIFAR100_BOUND_PARITY_ORACLE.md:25-29). Today the call
                // site is never reached on this path at all, confirmed with
                // NY_DENSEHEAD_TRACE=1.
                //
                // Sound: IBP bounds are a certified enclosure and are what the
                // sibling arms already install; the tightener only ever
                // intersects. Strictly weaker than a completed alpha bootstrap,
                // strictly better than abandoning the instance.
                let mut degraded = bootstrap_config.as_ref().clone();
                degraded.use_alpha_crown = false;
                degraded.use_forward_bounds = false;
                degraded.root_alpha_cap_secs = None;
                tracing::info!(
                    global_left_s = global_left.as_secs_f64(),
                    "#root-cap-degrade: root-alpha phase cap expired; falling back to certified \
                     IBP bootstrap so the dense-head tightener is still reached"
                );
                match compute_graph_bab_bootstrap(
                    graph,
                    input,
                    &degraded,
                    engine,
                    Some(bab_deadline),
                ) {
                    Ok(bootstrap) => bootstrap,
                    Err(_) => {
                        return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
                            lifecycle.timeout_result(),
                        )))
                    }
                }
            } else {
                // Invariant I2 (docs/DESIGN_MARGINAL_VALUE_SCHEDULER_2026-08-08.md):
                // reaching here means `classify_expiry` said the GLOBAL deadline
                // is spent, so the instance genuinely is over. That is the only
                // condition under which a phase-level DeadlineExceeded may be
                // reported as a whole-run timeout -- the confusion between the
                // two discarded 209.9 s of live ledger at four separate sites.
                return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
                    lifecycle.timeout_result(),
                )));
            }
        }
        Err(e) => return Err(e),
    };
    if let Some(optimizer_updates_completed) = bootstrap.phase_cap_optimizer_updates {
        info!(
            optimizer_updates_completed,
            verdict_authority = false,
            "#root-alpha-phase-checkpoint: entering ordinary root tightening and objective re-evaluation"
        );
    }

    // #root-alpha-margin (typed default OFF; `NY_ROOT_ALPHA_MARGIN` override):
    // carry the margin rows into the root α warmup so it can RANK its iterates by
    // the spec objective and hand back the best-scoring α instead of its last one.
    //
    // The warmup ascends `finite_lower_sum` over RAW output dims while these
    // properties are a conjunction of margin rows, and it keeps no best-α snapshot.
    // Historical 2026-07-26 pre-quarantine mechanism evidence on
    // CIFAR100_resnet_medium prop_idx_7704 moved 23/99 root rows after 1 iteration
    // to 0/99 after 10. This is not current sound-path performance evidence; see
    // `docs/CIFAR100_ROOT_ALPHA_DEGRADES_SPEC_BOUNDS_2026-07-26.md`.
    //
    // SOUND: selection only. The score never decides a verdict and never feeds a
    // bound; it only picks which α to keep, and every α ∈ [0,1] yields a valid bound.
    // Attached only to the bootstrap's alpha_config, which feeds the iterative root
    // warmup below. Reading the gate here (rather than in the loop alone) keeps the
    // config `None` when disarmed, so the root-cache identity is unchanged.
    // Kept for any LATER consumer of `bootstrap.alpha_config` (re-collections,
    // refresh passes). Reuses the value already built above so the ascent is
    // constructed once and both consumers see the identical rows.
    if let Some(spec) = root_alpha_spec {
        bootstrap.alpha_config.spec_ascent = Some(spec);
    }

    // STABILIZE-AND-FIX (#stabilize, dark `NY_STABILIZE=<budget_secs>`, default
    // OFF ⇒ byte-identical): spend a bounded root budget proving individual
    // unstable ReLU neurons stable (per-neuron α-CROWN backward tighten,
    // intersect-only into the stored pre-activation entry) and FIX the proven
    // ones. The tightened STORED entry (l≥0 / u≤0) is itself the proof artifact
    // that every relaxation consumer (constraints/backward/relu.rs, the GPU
    // Activation extraction) and every branching scan
    // (find_unstable_graph_neurons_*) reads, so fixed neurons lose their
    // triangle looseness and branching candidacy on EVERY descendant domain
    // with zero new trust surface. Runs BEFORE the MIP stash, the root
    // objective pass, and `build_graph_bab_setup`, so all of them inherit the
    // fixes. See `shared/stabilize.rs` for the loop + soundness invariants.
    // #root-phases: dispatch IN PLACE, one call where each block already sat.
    //
    // `ORDER` describes the sequence; it is not a site to execute it from. The
    // phases are separated by code that touches the same state, so a single
    // hoisted walk changes the program -- measured, bounds identical but
    // verdict timeout -> unknown. In-place dispatch preserves order exactly.
    let mut phase_out = super::root_phases::PhaseOutput::default();
    phase_out.merge(
        super::root_phases::RootTightenPhase::StabilizeAndFix.run_in_place(
            graph,
            input,
            objectives,
            engine,
            deadline,
            &mut bootstrap,
            &verifier.config,
            root_interm_cuda_factory_requested(&verifier.config),
            dd_zono.as_ref(),
        ),
    );
    phase_out.merge(
        super::root_phases::RootTightenPhase::DdZonoIntermIntersect.run_in_place(
            graph,
            input,
            objectives,
            engine,
            deadline,
            &mut bootstrap,
            &verifier.config,
            root_interm_cuda_factory_requested(&verifier.config),
            dd_zono.as_ref(),
        ),
    );

    // FC-head pre-activation tightening (#cifar100-fchead): the α-CROWN warmup
    // returns the forward-linear / IBP reference intermediate bounds unchanged
    // when fix_interm_bounds is set (the default deep-conv path). On deep conv
    // ResNets the dominant residual ReLU-relaxation slack at the output is
    // concentrated in the *dense* head pre-activation (cifar100 `Gemm_56`:
    // 2048→100, ~51/100 unstable, mean width 3.7 vs a <2%-unstable, tight conv
    // stack). Refine just those dense-fed ReLU pre-activations with a per-target
    // α-CROWN backward (reusing the warmup's optimized slopes) BEFORE the root
    // objective pass and the per-domain BaB setup consume the bounds, so both
    // benefit. The warmup deadline is already spent here, so this gets its own
    // small grace slice out of the remaining global budget (mirrors the root
    // spec pass grace). SOUND: `tighten_fc_head_preactivations` intersect-only —
    // it can only shrink a bound, never widen it, and every stored bound still
    // encloses the true reachable pre-activation set. Deadlines only schedule
    // work; on expiry each target keeps its sound reference bound.
    //
    // Gated OPT-IN (NY_FCHEAD_TIGHTEN=1): measured to close ~60% of the cifar100
    // resnet root-margin gap (helps deep classifiers with a loose dense FC head)
    // but adds ~1-2s at the BaB root on ANY net with a dense-fed unstable ReLU,
    // where it buys nothing — so it is OFF by default and enabled only where it
    // pays (the cifar100/tinyimagenet path / an explicit opt-in). It does not flip
    // cifar100 alone (the residual gap needs conv-stack tightening too).
    phase_out.merge(
        super::root_phases::RootTightenPhase::FcHeadTighten.run_in_place(
            graph,
            input,
            objectives,
            engine,
            deadline,
            &mut bootstrap,
            &verifier.config,
            root_interm_cuda_factory_requested(&verifier.config),
            dd_zono.as_ref(),
        ),
    );

    // Root intermediate-bound α tightening (#root-interm-alpha, dark
    // `NY_ROOT_INTERM_ALPHA=1`): the BROAD counterpart to NY_FCHEAD_TIGHTEN.
    // With `fix_interm_bounds=true` the α-CROWN warmup returns the heuristic-α
    // reference intermediate bounds unchanged — the α it optimized for the
    // output margin is never applied to the intermediate pre-activations, so
    // every crossing-ReLU triangle along the conv stack + FC head is relaxed
    // with heuristic (not optimized) slopes. auto_LiRPA instead optimizes the α
    // used to compute EACH intermediate bound. This pass recomputes ALL root
    // ReLU pre-activations (conv-stack BN/Add outputs AND the dense `Gemm_56`
    // head) with the warmup's OPTIMIZED α, BEFORE the root objective pass and
    // per-domain BaB setup consume the bounds (children inherit them as their
    // forward base), so the whole tree benefits. It measures whether the one
    // untested lever — optimized-α root intermediate bounds — moves the cifar100
    // worst-subdomain plateau. SOUND: `tighten_all_relu_preactivations` is
    // intersect-only (shrink a bound, never widen; α only tunes the ReLU lower
    // slope within the sound triangle); on deadline each target keeps its sound
    // reference bound. Default-OFF ⇒ byte-identical (no bound is ever touched).
    phase_out.merge(
        super::root_phases::RootTightenPhase::RootIntermAlpha.run_in_place(
            graph,
            input,
            objectives,
            engine,
            deadline,
            &mut bootstrap,
            &verifier.config,
            root_interm_cuda_factory_requested(&verifier.config),
            dd_zono.as_ref(),
        ),
    );

    // NY_ROOT_INTERM_ALPHA block's closing brace (before the next lever's comment).

    // Root JOINT per-target intermediate-bound α pass (#root-joint-interm-alpha,
    // dark `NY_ROOT_JOINT_INTERM_ALPHA=1`, default-OFF ⇒ byte-identical). The
    // auto_LiRPA `fix_intermediate_layer_bounds=False` root pass — the ONE α-family
    // variant never directly measured (docs/ROOT_JOINT_INTERM_ALPHA_PLAN.md §B).
    // Unlike the NY_ROOT_INTERM_ALPHA block above (which BORROWS the output-margin
    // warmup α and applies it ONCE per target — measured ZERO), this HOISTS the
    // per-target α′ ascent to the root: for each scoped target layer L it seeds
    // identity AT L's crossing rows and lets the gradient flow THROUGH L's own
    // intermediate-bound computation via the on-device joint adjoint
    // (`crown_joint_alpha_gradient_resident`), Adam-ascends the below-L α, scores
    // every iterate with the certified sound fold, and writes the element-wise
    // best box SHRINK-ONLY into `bootstrap.initial_node_bounds` HERE — before
    // `compute_root_objective_bounds` and the BaB-setup Arc — so the whole frozen
    // tree inherits it by pointer. Scope knobs:
    // NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM (contextual default 2048 ⇒ head + last
    // residual block; 32768 in the armed demand-ranked sound-GPU lane), _LAYERS
    // (comma-list of ReLU node names), _ITERS (default 100),
    // _LR (default 0.1), _SECS grace cap (default 30), _MAX_SEL (identity-seed
    // row cap, default 512). Finite-slice ascent additionally requires the typed
    // `NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT=1` policy and an exact
    // call-local bounded joint-adjoint capability; this multi-objective factory
    // route is not admitted unless both gates are exact. SOUND: every kept bound
    // comes from the sound fold; shrink-only intersect with per-element union
    // fallback; fail-closed on any refusal. Default-OFF ⇒ no bound is ever
    // touched.
    let root_joint_interm_tightened_targets = super::root_phases::root_joint_interm_alpha(
        graph,
        input,
        engine,
        deadline,
        verifier,
        objectives,
        &mut bootstrap,
    );

    // Resolve comprehensive-slot ownership once. The default-dark resident
    // owner claims this and the following wide slot without running here, then
    // carries its typed policy past sparse prerequisites to the dense-head
    // site. With that owner absent, the established comprehensive call and its
    // exact clean-decline ownership semantics execute unchanged.
    let (root_phase_resident_crown_policy, root_comprehensive_gpu_interm_tightened_targets) =
        phase_resident_or_comprehensive(root_phase_resident_crown_policy(), || {
            super::root_phases::comprehensive_gpu_interm_crown(
                graph,
                input,
                engine,
                deadline,
                &mut bootstrap,
                verifier,
                objectives,
            )
        });

    // One-target wide DEMANDED intermediate CROWN
    // (#root-wide-demanded-interm-crown, typed default-OFF). This is the dark
    // first slice for >=2,048-element demanded pre-activations. It ranks all
    // finite non-point width by downstream ReLU coverage, with crossing mass as
    // a secondary key, then asks the exact retained sound GPU capability for a
    // typed sweep. Local capability only: no backend-name check and no factory
    // retry; every unsupported or unauthorized path leaves the map untouched.
    let root_wide_demanded_interm_tightened_targets =
        comprehensive_gpu_or_legacy_wide(root_comprehensive_gpu_interm_tightened_targets, || {
            super::root_phases::wide_demanded_interm_crown(
                graph,
                input,
                engine,
                deadline,
                &mut bootstrap,
            )
        });

    // Typed sparse crossing-row intermediate CROWN (#root-sparse-interm-crown).
    // Unlike the research joint-α seam above, this production-shaped pass runs
    // only the certified BASE sound fold (zero optimization iterations), selects
    // convolutional/residual ReLU pre-activations structurally, excludes the
    // separately-owned dense head, and caps dimensions, selected rows, targets,
    // and wall time before allocation. Tightened boxes are shrink-only and are
    // inherited by the root objective and every BaB child. Default-OFF unless a
    // measured typed preset or the sealed force-on A/B gate enables it.
    let root_interm_factory_requested = root_interm_cuda_factory_requested(&verifier.config);
    let root_sparse_interm_tightened_targets = super::root_phases::sparse_interm_crown(
        graph,
        input,
        engine,
        deadline,
        root_interm_factory_requested,
        verifier,
        &mut bootstrap,
    );

    // Root CROWN-backward intermediate-bound INTERSECT (#root-crown-interm). At
    // the ROOT (before BaB), compute a SOUND heuristic-α CROWN BACKWARD box to
    // the input eps-box and INTERSECT it SHRINK-ONLY into the frozen
    // `initial_node_bounds`:
    //   l_new = max(l_fwd, l_crown),  u_new = min(u_fwd, u_crown).
    // SOUNDNESS: both boxes are sound enclosures of the reachable pre-activation
    // set, so their intersection is a sound enclosure (never drops a real point)
    // and can ONLY tighten. The tightened bounds are written into
    // `bootstrap.initial_node_bounds` HERE — before `compute_root_objective_bounds`
    // AND before the BaB-setup Arc (`build_graph_bab_setup`, below, which Arc-wraps
    // this same map) — so the root objective and EVERY BaB subdomain inherit the
    // tighter bounds by pointer; the CROWN pass runs ONCE at the root, never per
    // child. Production selection is structural: dense-fed ReLU pre-activations
    // only, armed by the typed benchmark preset. FAIL-CLOSED: the pass has its
    // own deadline, and any expiry/non-finite/disjoint/shape mismatch keeps the
    // forward-linear reference. Legacy env force-on/off and layer selection
    // remain available for A/B and rollback.
    // The critical-row candidate needs the typed dense-head stage to have been
    // selected, but it does not require that the intersection changed a scalar:
    // the bootstrap boxes were certified before this optional tightening pass.
    // A selected pass that reports zero changes is still a valid fresh-slope
    // starting point (and is the observed prop_1761 case).
    // #densehead-trace (DARK, NY_DENSEHEAD_TRACE=1): the dense-head tightener is
    // measured to close 97.82% of the ny-vs-abc root head-width gap in 0.5s
    // (docs/CIFAR100_BOUND_PARITY_ORACLE.md:25-29) yet emits nothing in a
    // shipped run. Distinguish "call site not reached" from "policy declined"
    // from "budget declined" -- all three are silent otherwise.
    if std::env::var("NY_DENSEHEAD_TRACE").is_ok_and(|v| v == "1") {
        let pol = root_crown_interm_policy_from_env(&verifier.config);
        let now = std::time::Instant::now();
        eprintln!(
            "[densehead] reached=yes preset_dense_head={} policy={} max_secs={:?}              global_remaining_s={:?} slice={:?}",
            verifier.config.root_crown_interm_dense_head,
            pol.is_some(),
            pol.as_ref().map(|p| p.max_secs),
            deadline.map(|d| d.saturating_duration_since(now).as_secs_f64()),
            pol.as_ref()
                .and_then(|p| bounded_root_crown_interm_deadline(now, deadline, p.max_secs))
                .map(|d| d.saturating_duration_since(now).as_secs_f64()),
        );
    }
    let dense_head_out = if let Some(policy) = root_phase_resident_crown_policy {
        let resident = super::root_phases::RootTightenPhase::PhaseResidentCrown
            .run_phase_resident_in_place(graph, input, engine, deadline, &mut bootstrap, policy);
        if resident.permits_legacy_dense_fallback() {
            super::root_phases::RootTightenPhase::DenseHeadTighten.run_in_place(
                graph,
                input,
                objectives,
                engine,
                deadline,
                &mut bootstrap,
                &verifier.config,
                root_interm_factory_requested,
                dd_zono.as_ref(),
            )
        } else {
            resident.output()
        }
    } else {
        super::root_phases::RootTightenPhase::DenseHeadTighten.run_in_place(
            graph,
            input,
            objectives,
            engine,
            deadline,
            &mut bootstrap,
            &verifier.config,
            root_interm_factory_requested,
            dd_zono.as_ref(),
        )
    };
    // #root-phases producer channel. Both legacy dense element counts and the
    // resident transaction's target count flow through one PhaseOutput, which
    // is the sole input to every later stale-box decision.
    let root_dense_head_stage_selected = dense_head_out.dense_head_stage_selected;
    let root_intermediate_bounds_changed = root_intermediate_tightening_changed(
        dense_head_out,
        root_comprehensive_gpu_interm_tightened_targets,
        root_wide_demanded_interm_tightened_targets,
        root_sparse_interm_tightened_targets,
        root_joint_interm_tightened_targets,
    );
    // Graph-MIP stash (FIX 1, `docs/GRAPH_MIP_LEAF_SOLVER.md`): the relational
    // multi-objective lane computes its per-property bounds HERE (not at the
    // ny-cli per-constraint precompute), so this is where the Graph-MIP
    // escalation's reuse mailbox must be filled — otherwise the escalation
    // falls back to a deadline-truncated recompute whose LOOSE bounds inflate
    // the unstable-ReLU eligibility count. Disabled when whole-net Graph-MIP
    // is explicitly off or the category requests no MIP reservation; the leaf
    // oracle consumes child bounds directly and remains independent.
    crate::beta_crown::graph_mip_leaf::stash_root_bounds_for_mip(
        graph,
        input,
        &verifier.config.phase_budget,
        &bootstrap.initial_node_bounds,
    );
    // DIAGNOSTIC (NY_LPOPT_DUMP=<path>): dump the EXACT root state feeding every BaB
    // subdomain — the input eps-box + the full per-node pre-activation `[l,u]`
    // (`bootstrap.initial_node_bounds`, AFTER the optional CROWN-interm tighten) +
    // the ReLU→pre-activation node-name map. This is the ground-truth data needed to
    // rebuild NY's OWN triangle-relaxation LP off-line (p*_LP) and check whether NY's
    // α/β-CROWN reaches its own relaxation's LP optimum. Read-only / print-only;
    // never mutates the bootstrap or any verdict. Default-OFF ⇒ byte-identical.
    if let Ok(path) = std::env::var("NY_LPOPT_DUMP") {
        run_lpopt_dump(graph, input, &bootstrap, &path);
    }
    // #phase-telemetry (dark, NY_PHASE_TELEMETRY=1, print-only): bracket the
    // root objective evaluation. A start without an end in a log means the
    // phase timed out (the DeadlineExceeded arm) or errored.
    crate::phase_telemetry::phase_marker("root-objective start");
    let RootObjectiveEvaluation {
        initial_output,
        mut initial_obj_bounds,
        root_spec_cache,
        root_spec_cache_active_indices,
        root_alpha_override,
    } = match compute_root_objective_bounds(
        verifier,
        graph,
        input,
        objectives,
        thresholds,
        conjunctive,
        engine,
        &bootstrap,
        deadline,
        root_intermediate_bounds_changed,
        root_dense_head_stage_selected,
    ) {
        Ok(evaluation) => evaluation,
        Err(ny_core::NyError::DeadlineExceeded(_)) => {
            // Invariant I2 (#phase-yield). This pass runs under ROOT_SPEC_GRACE
            // (3 s on this preset), not under the BaB deadline. Reporting its
            // expiry as a whole-instance Timeout is the defect that discarded
            // 209.9 s of live ledger elsewhere in this function.
            //
            // A spent GLOBAL deadline is a genuine timeout. A spent phase grace
            // with the instance still live is an Unknown: it is the honest
            // verdict (we did not finish this pass), it does not claim the clock
            // ran out, and it leaves the downstream escalation lanes -- which
            // admit on Unknown|Timeout alike -- their remaining budget.
            let status = match ny_core::phase_yield::classify_expiry(
                std::time::Instant::now(),
                None,
                Some(bab_deadline),
            ) {
                ny_core::phase_yield::Expiry::Global => {
                    return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
                        lifecycle.timeout_result(),
                    )))
                }
                ny_core::phase_yield::Expiry::PhaseOnly => BabVerificationStatus::Unknown {
                    reason: "root objective pass exceeded its phase grace while the \
                                 instance deadline was still live (#phase-yield I2)"
                        .to_string(),
                },
            };
            return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
                lifecycle.build_result(status),
            )));
        }
        Err(e) => return Err(e),
    };
    let fresh_atomic_root_alpha_installed = root_alpha_override.is_some();
    if let Some(alpha_state) = root_alpha_override {
        // This optional value can only be minted by the exact-dark
        // NY_ROOT_ALPHA_CUDA_ROWS + NY_ROOT_ALPHA_CUDA_MARGIN_STEP path after
        // a complete alpha1 C evaluation won whole-candidate selection.
        bootstrap.root_alpha_state = Some(alpha_state);
        info!(
            bab_warm_start_enabled = verifier.config.beta_iterations > 0,
            "Atomic CUDA margin-alpha: installed explicitly selected alpha1 state for downstream root processing"
        );
    }
    crate::phase_telemetry::phase_marker("root-objective end");

    // #dd-zonotope INTERSECT: the certified zonotope margin and the CROWN
    // margin are two sound enclosures of the same objective values, so keeping
    // the tighter side of each is sound and can only raise a lower bound /
    // lower an upper bound. A refusal above leaves `dd_zono` as `None` and this
    // block is inert.
    if let Some(result) = dd_zono.as_ref() {
        let tightened = intersect_objective_bounds(&mut initial_obj_bounds, &result.margin);
        if tightened > 0 {
            info!(
                "#dd-zonotope: intersected {}/{} root objective bound(s)",
                tightened,
                initial_obj_bounds.len()
            );
        }
    }

    // Multi-neuron (k-ReLU) ROOT injection (increment 3, NY_MULTINEURON=1,
    // default-OFF). Runs the objective backward with sound coupling facets
    // injected at the head ReLU (§2.2) and combines the injected margin lower
    // bound with the baseline by a per-objective sound MAX — it can only RAISE
    // the certified lower bound, feeding both the root verdict and BaB. Byte-
    // identical when the gate is off. Given its own bounded grace slice out of
    // the remaining global budget (like the FC-head lever), so BaB keeps room.
    if crate::multineuron::root_inject::enabled() {
        let now = std::time::Instant::now();
        let grace_cap = std::time::Duration::from_secs(
            std::env::var("NY_MULTINEURON_GRACE_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(30),
        );
        let global_remaining = deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or(grace_cap);
        let slice = grace_cap.min(global_remaining.mul_f32(0.6));
        // Measurement override (NY_MULTINEURON_NODEADLINE=1): give the injected
        // backwards NO deadline so they complete against the tight acrown
        // intermediates (the ~-0.829 CPU backward) instead of IBP-falling-back
        // to the loose IBP intermediates (~-20). Deadlines only schedule work, so
        // this is sound; it may overrun the scored budget and is for A/B only.
        let mn_deadline = if matches!(
            std::env::var("NY_MULTINEURON_NODEADLINE").ok().as_deref(),
            Some("1")
        ) {
            None
        } else {
            Some(now + slice.max(std::time::Duration::from_secs(2)))
        };
        initial_obj_bounds = crate::multineuron::root_inject::tighten_root_objective_bounds(
            graph,
            input,
            objectives,
            engine,
            &bootstrap.initial_node_bounds,
            bootstrap.root_alpha_state.as_ref(),
            &initial_obj_bounds,
            mn_deadline,
        );
    }

    // STEM-RESIDENT research implementation. `stem_enabled()` is production-
    // authority quarantined until the facet support, fold reduction error, and
    // target/model binding have checker-backed certificates. This block is
    // therefore unreachable from environment requests today.
    if crate::multineuron::root_inject::stem_enabled() {
        let now = std::time::Instant::now();
        let grace_cap = std::time::Duration::from_secs(
            std::env::var("NY_MULTINEURON_STEM_GRACE_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(30),
        );
        let global_remaining = deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or(grace_cap);
        let stem_deadline = if matches!(
            std::env::var("NY_MULTINEURON_NODEADLINE").ok().as_deref(),
            Some("1")
        ) {
            None
        } else {
            Some(
                now + grace_cap
                    .min(global_remaining.mul_f32(0.6))
                    .max(std::time::Duration::from_secs(2)),
            )
        };
        initial_obj_bounds =
            crate::multineuron::root_inject::tighten_root_objective_bounds_stem_resident(
                graph,
                input,
                objectives,
                engine,
                &bootstrap.initial_node_bounds,
                bootstrap.root_alpha_state.as_ref(),
                &initial_obj_bounds,
                stem_deadline,
            );
    }

    // #f64-tail-root: certified f64 replay of the still-unresolved objective rows.
    //
    // ny's f32 CROWN carries an a-priori certified coefficient error that is
    // RE-BOUNDED at every layer and multiplied by the kernel norm, so on a deep
    // conv DAG it compounds geometrically — measured with NY_ERR_SPLIT, 93-100%
    // of the emitted error is that carry. A single end-to-end f64 replay has no
    // carry: its error is measured once, at the end. So a row whose f32 lower
    // bound sits just below its threshold can still be genuinely refutable, and
    // the replay is exactly the instrument that shows it.
    //
    // SOUNDNESS: `f64_tail_box_attempt` returns `true` ONLY on a full
    // certified-outward f64 refutation of every row passed (its `Verified` is the
    // only verdict-changing outcome; Unsupported/NotVerified change nothing). We
    // therefore only ever RAISE a lower bound to just above its threshold, and
    // only for rows the replay certifies — `max(lb_f32, lb_f64)`, downgrade-only.
    // A row we cannot certify keeps its f32 bound untouched, so this can add
    // proofs but never remove or weaken one.
    //
    // MEASURED REACH: verified INERT on cifar100_2024 CIFAR100_resnet_medium at
    // both 100s and 900s budgets — an instrumented build printed nothing here,
    // because `compute_root_objective_bounds` above exhausts the whole budget
    // and returns via its DeadlineExceeded arm long before this point. That is a
    // statement about that benchmark's root cost, not about this hook: it
    // engages on any instance whose root objective evaluation completes with
    // rows still unresolved.
    if crate::network::graph_crown_f64_tail::f64_tail_enabled()
        && deadline.is_none_or(|d| std::time::Instant::now() < d)
    {
        let unresolved: Vec<usize> = initial_obj_bounds
            .iter()
            .zip(thresholds.iter())
            .enumerate()
            .filter(|(_, ((lower, _), threshold))| *lower <= **threshold)
            .map(|(i, _)| i)
            .collect();
        if !unresolved.is_empty() {
            let output_dim = initial_output.len();
            if let Some(full_spec) = build_spec_matrix(objectives) {
                let mut recovered = 0usize;
                for row in unresolved {
                    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                        break;
                    }
                    if full_spec.ncols() != output_dim || row >= full_spec.nrows() {
                        continue;
                    }
                    let spec_row = full_spec.row(row).insert_axis(ndarray::Axis(0)).to_owned();
                    let ths = vec![thresholds[row]];
                    if crate::network::f64_tail_box_attempt(
                        graph,
                        input,
                        &spec_row,
                        &ths,
                        &[1],
                        engine,
                        deadline,
                    ) {
                        // Certified: margin_lb > threshold. Record the proven fact
                        // without inventing a magnitude the replay did not report.
                        let t = thresholds[row];
                        let raised = next_up_f32(t);
                        if raised > initial_obj_bounds[row].0 {
                            initial_obj_bounds[row].0 = raised;
                            recovered += 1;
                        }
                    }
                }
                if recovered > 0 {
                    info!(
                        "#f64-tail-root: certified f64 replay recovered {recovered} objective row(s) \
                         the f32 backward left unresolved"
                    );
                }
            }
        }
    }

    // #mn-head-facet increment 1 (dark, NY_MN_HEAD_FACET=1, default-OFF). Build the
    // HEAD k-ReLU coupling-facet research construction. These requests may
    // populate the shared registry for offline inspection, but ny-core's
    // proof-path reader is hard-quarantined and never exposes an entry to the
    // CPU f64 recovery. Thus neither environment request can change root,
    // child, or verdict-bearing bounds.
    if crate::multineuron::root_inject::head_facet_enabled()
        || crate::multineuron::root_inject::head_f64_certified_measure_enabled()
    {
        crate::multineuron::root_inject::install_head_f64_fold(
            graph,
            input,
            objectives,
            &bootstrap.initial_node_bounds,
            bootstrap.root_alpha_state.as_ref(),
            engine,
        );
    }

    // #mn-head-resident (dark, NY_MN_HEAD_RESIDENT=1, default-OFF, byte-identical
    // when unset). The UNMASKED head lever: thread the HEAD coupling facets into
    // the OPTIMIZED resident GPU backward, RETARGETED from the stem to the head
    // act (fold index 0) — so the facet rides the tight GPU baseline itself rather
    // than the (masked) CPU f64 recovery. Single-global sound (root facet valid on
    // every subdomain); GUARD1 refuses any pool-node vs head-target mismatch. Sound
    // per-objective MAX with the baseline: it can only RAISE the certified lower
    // bound (INV-A: β=0 == baseline; INV-B: β>0 non-decreasing).
    if crate::multineuron::root_inject::head_resident_enabled() {
        let now = std::time::Instant::now();
        let grace_cap = std::time::Duration::from_secs(
            std::env::var("NY_MN_HEAD_RESIDENT_GRACE_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(30),
        );
        let global_remaining = deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or(grace_cap);
        let head_deadline = if matches!(
            std::env::var("NY_MULTINEURON_NODEADLINE").ok().as_deref(),
            Some("1")
        ) {
            None
        } else {
            Some(
                now + grace_cap
                    .min(global_remaining.mul_f32(0.6))
                    .max(std::time::Duration::from_secs(2)),
            )
        };
        initial_obj_bounds =
            crate::multineuron::root_inject::tighten_root_objective_bounds_head_resident(
                graph,
                input,
                objectives,
                engine,
                &bootstrap.initial_node_bounds,
                bootstrap.root_alpha_state.as_ref(),
                &initial_obj_bounds,
                head_deadline,
            );
    }

    // One binding-margin alpha step on the exact state that will become the
    // BaB root (dark, exact `NY_ROOT_CRITICAL_GPU_ALPHA=1`).  Planning happens
    // after every root tightening/intersection, and admits only the selected
    // dense-head, disjunctive, non-truncated, single-unresolved-row surface.
    //
    // The private two-second slice is cooperative: it cannot preempt an
    // in-flight CUDA kernel, but every heavyweight boundary and final
    // publication polls the deadline. The initial and post-Adam direct-C
    // results remain indivisible bound/state pairs; paired best-of keeps the
    // initial pair when Adam regresses. Any refusal returns `None`, preserving
    // the ordinary root-state and per-disjunct path byte-for-byte.
    let critical_gpu_alpha_artifacts =
        with_critical_gpu_alpha_gate(critical_gpu_alpha_enabled, || {
            let started_at = std::time::Instant::now();
            let authority_deadline = Some(
                deadline
                    .map(|global| bab_deadline.min(global))
                    .unwrap_or(bab_deadline),
            );
            // Nested dark K-row lane. The active-set variable is read only when
            // the existing LR bracket is already armed. K=1 falls through to
            // the sealed scalar block below without changing its plan,
            // arithmetic, deadline, executor, or publication.
            let bracket_enabled = critical_gpu_alpha_lr_bracket_enabled();
            if root_critical_gpu_alpha_active_set_enabled_if_bracket(bracket_enabled) {
                let classification =
                    match classify_complete_active_set_gpu_alpha(&initial_obj_bounds, thresholds) {
                        Ok(classification) => classification,
                        Err(reason) => {
                            let rows = match reason {
                                ActiveSetGpuAlphaRootRefusal::Classification(
                                    ActiveSetGpuAlphaRefusal::TooManyUnresolvedRows {
                                        count, ..
                                    },
                                ) => Some(count),
                                _ => None,
                            };
                            emit_active_set_gpu_alpha_telemetry(
                                ActiveSetGpuAlphaTelemetry::PlanRefused { rows, reason },
                            );
                            return None;
                        }
                    };
                match classification {
                    ActiveSetGpuAlphaClassification::DelegateSealedCriticalRow(delegation) => {
                        // The scalar planner below independently derives and
                        // validates this row; this assertion cannot affect
                        // release behavior.
                        debug_assert!(delegation.source_row_index() < initial_obj_bounds.len());
                    }
                    ActiveSetGpuAlphaClassification::Optimize(mut active_plan) => {
                        if let Err(reason) = validate_active_set_root_surface(
                            root_dense_head_stage_selected,
                            conjunctive,
                            verifier.config.crown_backward_layers,
                        ) {
                            emit_active_set_gpu_alpha_telemetry(
                                ActiveSetGpuAlphaTelemetry::PlanRefused {
                                    rows: Some(active_plan.len()),
                                    reason,
                                },
                            );
                            return None;
                        }
                        let hard_deadline = match root_critical_gpu_deadline_with_runtime(
                            started_at,
                            authority_deadline,
                            ROOT_CRITICAL_GPU_ALPHA_MAX_RUNTIME,
                        ) {
                            Some(deadline) => deadline,
                            None => {
                                let reason = ActiveSetGpuAlphaRootRefusal::Plan(
                                    CriticalGpuSpecRefusal::InsufficientHeadroom,
                                );
                                emit_active_set_gpu_alpha_telemetry(
                                    ActiveSetGpuAlphaTelemetry::PlanRefused {
                                        rows: Some(active_plan.len()),
                                        reason,
                                    },
                                );
                                return None;
                            }
                        };
                        let full_spec_matrix = match build_spec_matrix(objectives) {
                            Some(matrix) => matrix,
                            None => {
                                let reason = ActiveSetGpuAlphaRootRefusal::Plan(
                                    CriticalGpuSpecRefusal::InvalidInput,
                                );
                                emit_active_set_gpu_alpha_telemetry(
                                    ActiveSetGpuAlphaTelemetry::PlanRefused {
                                        rows: Some(active_plan.len()),
                                        reason,
                                    },
                                );
                                return None;
                            }
                        };

                        // This setup/state is retained verbatim only if atomic
                        // K-vector/state publication succeeds.
                        let graph_setup =
                            build_graph_bab_setup(graph, &bootstrap.initial_node_bounds);
                        let root_history = GraphSplitHistory::new();
                        let initial_alpha_state = build_root_alpha_state(
                            graph,
                            input,
                            &root_history,
                            &graph_setup.initial_node_bounds_arc,
                            if root_intermediate_bounds_changed {
                                None
                            } else {
                                bootstrap.root_alpha_state.as_ref()
                            },
                            verifier.config.beta_iterations > 0,
                        );
                        let routed = run_active_set_gpu_alpha_candidate(
                            &mut active_plan,
                            graph,
                            input,
                            &bootstrap.initial_node_bounds,
                            engine,
                            &full_spec_matrix,
                            hard_deadline,
                            &initial_alpha_state,
                            &verifier.config.adaptive_config,
                            verifier.config.alpha_lr,
                        );
                        let output = match routed.result {
                            Ok(output) => output,
                            Err(reason) => {
                                emit_active_set_gpu_alpha_telemetry(
                                    ActiveSetGpuAlphaTelemetry::CandidateRefused {
                                        backend: routed.backend,
                                        rows: active_plan.len(),
                                        reason,
                                    },
                                );
                                return None;
                            }
                        };
                        for trace in output.candidate_traces() {
                            emit_active_set_gpu_alpha_telemetry(
                                ActiveSetGpuAlphaTelemetry::Candidate {
                                    rows: active_plan.len(),
                                    trace,
                                },
                            );
                        }
                        let pair = match build_active_set_gpu_alpha_publication(
                            &initial_obj_bounds,
                            &active_plan,
                            output,
                            hard_deadline,
                        ) {
                            Ok(pair) => pair,
                            Err(reason) => {
                                emit_active_set_gpu_alpha_telemetry(
                                    ActiveSetGpuAlphaTelemetry::CandidateRefused {
                                        backend: routed.backend,
                                        rows: active_plan.len(),
                                        reason,
                                    },
                                );
                                return None;
                            }
                        };
                        let tag = pair
                            .active_set_transport_tag
                            .as_ref()
                            .expect("accepted active-set publication is tagged");
                        emit_active_set_gpu_alpha_telemetry(ActiveSetGpuAlphaTelemetry::Accepted {
                            backend: routed.backend,
                            tag,
                        });
                        info!(
                            status = "accepted",
                            backend = routed.backend.telemetry_name(),
                            rows = tag.rows.len(),
                            elapsed_ms = started_at.elapsed().as_millis() as u64,
                            certified = tag.score.rows_certified(),
                            min_slack = tag.score.min_slack(),
                            negative_slack_sum = tag.score.negative_slack_sum(),
                            alpha_params = tag.state_identity.parameter_count(),
                            alpha_fingerprint = tag.state_identity.fingerprint(),
                            pair_fingerprint = tag.pair_fingerprint.value(),
                            cache_published = false,
                            "Active-set state-paired GPU alpha root candidate"
                        );
                        // Read the subordinate gate only after the parent
                        // active-set lane has produced an authoritative pair.
                        // Gate-off returns that exact pair without further
                        // planning, environment reads, or scalar work.
                        if !root_critical_gpu_alpha_active_set_cascade_enabled_if_active(true) {
                            return Some(CriticalGpuAlphaRootArtifacts { graph_setup, pair });
                        }

                        // The scalar retry receives the original active-set
                        // authority boundary verbatim. No `now + runtime`
                        // deadline, reserve reset, or fresh factory budget is
                        // constructed here.
                        let cascade_plan = build_active_set_scalar_cascade_plan(
                            &pair.objective_bounds,
                            &full_spec_matrix,
                            thresholds,
                            hard_deadline,
                            std::time::Instant::now(),
                        );
                        let cascade_plan = match cascade_plan {
                            Ok(plan) => plan,
                            Err(reason) => {
                                emit_active_set_scalar_cascade_telemetry(
                                    ActiveSetScalarCascadeTelemetry::Refused {
                                        backend: None,
                                        row: None,
                                        reason,
                                    },
                                );
                                let (pair, retained_reason) =
                                    retain_active_set_pair_on_cascade_refusal(pair, Err(reason));
                                debug_assert_eq!(retained_reason, Some(reason));
                                return Some(CriticalGpuAlphaRootArtifacts { graph_setup, pair });
                            }
                        };
                        let cascade_backend = select_critical_gpu_spec_backend(engine);
                        emit_active_set_scalar_cascade_telemetry(
                            ActiveSetScalarCascadeTelemetry::BackendSelected {
                                backend: cascade_backend,
                                row: cascade_plan.row_index,
                            },
                        );
                        let scalar_routed = run_critical_gpu_alpha_candidate(
                            graph,
                            input,
                            &bootstrap.initial_node_bounds,
                            engine,
                            &cascade_plan,
                            &pair.alpha_state,
                            &verifier.config.adaptive_config,
                            Some(verifier.config.alpha_lr),
                        );
                        debug_assert_eq!(scalar_routed.backend, cascade_backend);
                        let cascade_result = match scalar_routed.result {
                            Ok(evaluation) => {
                                if let Some(provenance) = evaluation.search_provenance.as_ref() {
                                    for candidate in &provenance.candidates {
                                        emit_active_set_scalar_cascade_telemetry(
                                            ActiveSetScalarCascadeTelemetry::Candidate {
                                                row: cascade_plan.row_index,
                                                candidate,
                                            },
                                        );
                                    }
                                }
                                match build_critical_gpu_alpha_publication(
                                    &pair.objective_bounds,
                                    &cascade_plan,
                                    evaluation,
                                ) {
                                    Ok(scalar_publication) => {
                                        build_active_set_scalar_cascade_publication_with_clock(
                                            &pair,
                                            scalar_publication,
                                            &cascade_plan,
                                            thresholds,
                                            hard_deadline,
                                            std::time::Instant::now,
                                        )
                                    }
                                    Err(reason) => {
                                        Err(ActiveSetScalarCascadeRefusal::Scalar(reason))
                                    }
                                }
                            }
                            Err(reason) => Err(ActiveSetScalarCascadeRefusal::Scalar(reason)),
                        };
                        let (pair, retained_reason) =
                            retain_active_set_pair_on_cascade_refusal(pair, cascade_result);
                        if let Some(reason) = retained_reason {
                            emit_active_set_scalar_cascade_telemetry(
                                ActiveSetScalarCascadeTelemetry::Refused {
                                    backend: Some(cascade_backend),
                                    row: Some(cascade_plan.row_index),
                                    reason,
                                },
                            );
                            info!(
                                status = "refused",
                                backend = cascade_backend.telemetry_name(),
                                row = cascade_plan.row_index,
                                reason = reason.telemetry_reason(),
                                elapsed_ms = started_at.elapsed().as_millis() as u64,
                                retained = "active-set",
                                "Active-set to scalar GPU alpha root cascade"
                            );
                            return Some(CriticalGpuAlphaRootArtifacts { graph_setup, pair });
                        }
                        let cascade_tag = pair
                            .active_set_scalar_cascade_transport_tag
                            .as_ref()
                            .expect("accepted cascade publication is tagged");
                        let scalar_tag = pair
                            .transport_tag
                            .as_ref()
                            .expect("accepted cascade retains scalar state provenance");
                        emit_active_set_scalar_cascade_telemetry(
                            ActiveSetScalarCascadeTelemetry::Accepted {
                                backend: cascade_backend,
                                tag: cascade_tag,
                                scalar: scalar_tag,
                            },
                        );
                        info!(
                            status = "accepted",
                            backend = cascade_backend.telemetry_name(),
                            row = cascade_tag.survivor_row,
                            elapsed_ms = started_at.elapsed().as_millis() as u64,
                            merged_lower = scalar_tag.merged_lower,
                            lift = scalar_tag.merged_lower - scalar_tag.historical_lower,
                            active_state_fingerprint =
                                cascade_tag.active_set.state_identity.fingerprint(),
                            final_state_fingerprint = cascade_tag.final_state_identity.fingerprint,
                            bounds_fingerprint = cascade_tag.published_bounds_fingerprint,
                            cache_published = false,
                            "Active-set to scalar certified-intersection GPU alpha root cascade"
                        );
                        return Some(CriticalGpuAlphaRootArtifacts { graph_setup, pair });
                    }
                }
            }
            let full_spec_matrix = match build_spec_matrix(objectives) {
                Some(matrix) => matrix,
                None => {
                    let reason =
                        CriticalGpuAlphaRefusal::Plan(CriticalGpuSpecRefusal::InvalidInput);
                    emit_critical_gpu_alpha_telemetry(CriticalGpuAlphaTelemetry::PlanRefused {
                        reason,
                    });
                    return None;
                }
            };
            let plan = match build_critical_gpu_spec_plan_with_runtime(
                root_dense_head_stage_selected,
                conjunctive,
                verifier.config.crown_backward_layers,
                &initial_obj_bounds,
                &full_spec_matrix,
                thresholds,
                started_at,
                authority_deadline,
                ROOT_CRITICAL_GPU_ALPHA_MAX_RUNTIME,
            ) {
                Ok(plan) => plan,
                Err(plan_reason) => {
                    let reason = CriticalGpuAlphaRefusal::Plan(plan_reason);
                    emit_critical_gpu_alpha_telemetry(CriticalGpuAlphaTelemetry::PlanRefused {
                        reason,
                    });
                    info!(
                        status = "refused",
                        reason = reason.telemetry_reason(),
                        "Critical-row state-paired GPU alpha root candidate"
                    );
                    return None;
                }
            };

            // This setup/state is not a probe-only reconstruction: it is kept
            // and installed verbatim on the eventual root domain.
            let graph_setup = build_graph_bab_setup(graph, &bootstrap.initial_node_bounds);
            let root_history = GraphSplitHistory::new();
            let initial_alpha_state = build_root_alpha_state(
                graph,
                input,
                &root_history,
                &graph_setup.initial_node_bounds_arc,
                if root_intermediate_bounds_changed {
                    None
                } else {
                    bootstrap.root_alpha_state.as_ref()
                },
                verifier.config.beta_iterations > 0,
            );
            let baseline_pair = CriticalGpuAlphaRootPair {
                objective_bounds: initial_obj_bounds.clone(),
                alpha_state: initial_alpha_state,
                transport_tag: None,
                active_set_transport_tag: None,
                active_set_scalar_cascade_transport_tag: None,
            };
            // This nested gate is inspected only after the parent alpha lane
            // has admitted the request. `None` routes through the sealed
            // one-step function without changing its deadline or arithmetic.
            let bracket_base_lr =
                critical_gpu_alpha_lr_bracket_enabled().then_some(verifier.config.alpha_lr);
            let routed = run_critical_gpu_alpha_candidate(
                graph,
                input,
                &bootstrap.initial_node_bounds,
                engine,
                &plan,
                &baseline_pair.alpha_state,
                &verifier.config.adaptive_config,
                bracket_base_lr,
            );
            let pair = match routed.result {
                Ok(evaluation) => {
                    if let Some(provenance) = evaluation.search_provenance.as_ref() {
                        for candidate in &provenance.candidates {
                            emit_critical_gpu_alpha_telemetry(
                                CriticalGpuAlphaTelemetry::BracketCandidate {
                                    row: plan.row_index,
                                    candidate,
                                },
                            );
                        }
                    }
                    match build_critical_gpu_alpha_publication(
                        &baseline_pair.objective_bounds,
                        &plan,
                        evaluation,
                    ) {
                        Ok(publication) => {
                            let tag = publication
                                .pair
                                .transport_tag
                                .as_ref()
                                .expect("accepted alpha publication is tagged");
                            emit_critical_gpu_alpha_telemetry(
                                CriticalGpuAlphaTelemetry::Accepted {
                                    backend: routed.backend,
                                    tag,
                                },
                            );
                            info!(
                                status = "accepted",
                                backend = routed.backend.telemetry_name(),
                                row = tag.row_index,
                                elapsed_ms = started_at.elapsed().as_millis() as u64,
                                historical_lower = publication.accepted.historical_lower,
                                initial_lower = tag.initial_lower,
                                final_lower = tag.final_lower,
                                selected = tag.selected_pair.telemetry_name(),
                                merged_lower = publication.accepted.merged_lower,
                                lift = publication.accepted.merged_lower
                                    - publication.accepted.historical_lower,
                                alpha_params = tag.state_identity.parameter_count,
                                alpha_fingerprint = tag.state_identity.fingerprint,
                                cache_published = false,
                                "Critical-row state-paired GPU alpha root candidate"
                            );
                            publication.pair
                        }
                        Err(reason) => {
                            emit_critical_gpu_alpha_telemetry(
                                CriticalGpuAlphaTelemetry::CandidateRefused {
                                    backend: routed.backend,
                                    row: plan.row_index,
                                    reason,
                                },
                            );
                            info!(
                                status = "refused",
                                backend = routed.backend.telemetry_name(),
                                row = plan.row_index,
                                reason = reason.telemetry_reason(),
                                elapsed_ms = started_at.elapsed().as_millis() as u64,
                                "Critical-row state-paired GPU alpha root publication"
                            );
                            return None;
                        }
                    }
                }
                Err(reason) => {
                    emit_critical_gpu_alpha_telemetry(
                        CriticalGpuAlphaTelemetry::CandidateRefused {
                            backend: routed.backend,
                            row: plan.row_index,
                            reason,
                        },
                    );
                    info!(
                        status = "refused",
                        backend = routed.backend.telemetry_name(),
                        row = plan.row_index,
                        reason = reason.telemetry_reason(),
                        elapsed_ms = started_at.elapsed().as_millis() as u64,
                        "Critical-row state-paired GPU alpha root candidate"
                    );
                    return None;
                }
            };
            Some(CriticalGpuAlphaRootArtifacts { graph_setup, pair })
        })
        .flatten();

    let mut prebuilt_graph_setup = None;
    let mut prebuilt_root_alpha_state = None;
    let mut critical_gpu_alpha_transport_tag = None;
    let mut active_set_gpu_alpha_transport_tag = None;
    let mut active_set_scalar_cascade_transport_tag = None;
    if let Some(artifacts) = critical_gpu_alpha_artifacts {
        let CriticalGpuAlphaRootArtifacts { graph_setup, pair } = artifacts;
        let CriticalGpuAlphaRootPair {
            objective_bounds,
            alpha_state,
            transport_tag,
            active_set_transport_tag,
            active_set_scalar_cascade_transport_tag: cascade_transport_tag,
        } = pair;
        // MERGE, do not REPLACE (#root-bounds-last-writer-wins, 2026-08-03).
        // This candidate used to overwrite the whole vector, silently
        // discarding every raise the earlier root phases had certified —
        // measured on cifar100 prop_idx_7500: the row-conditional INVPROP lane
        // proved 8/8 of its rows outright (certified contradictions, before
        // -2.18..-6.49) and the published bounds showed no trace of it,
        // because this line ran afterwards. The f64-tail and multineuron
        // phases feed the same vector and were equally exposed.
        //
        // Both operands are certified enclosures of the SAME margins, so
        // keeping the tighter side of each row is sound (identical argument to
        // `intersect_objective_bounds`). Length drift falls back to the
        // historical replacement rather than publishing a mixed-arity vector.
        if initial_obj_bounds.len() == objective_bounds.len() {
            let mut kept = 0usize;
            for (slot, candidate) in initial_obj_bounds.iter_mut().zip(objective_bounds.iter()) {
                let (lo, hi) = *candidate;
                if lo.is_finite() && hi.is_finite() && lo <= hi {
                    if lo > slot.0 {
                        slot.0 = lo;
                    }
                    if hi < slot.1 {
                        slot.1 = hi;
                    }
                } else {
                    kept += 1;
                }
            }
            if kept > 0 {
                debug!(
                    skipped = kept,
                    "critical-row GPU alpha root pair: non-finite rows kept from the prior bounds"
                );
            }
        } else {
            initial_obj_bounds = objective_bounds;
        }
        prebuilt_graph_setup = Some(graph_setup);
        prebuilt_root_alpha_state = Some(alpha_state);
        critical_gpu_alpha_transport_tag = transport_tag;
        active_set_gpu_alpha_transport_tag = active_set_transport_tag;
        active_set_scalar_cascade_transport_tag = cascade_transport_tag;
    }

    // A #row-conditional-invprop SECOND PASS was tried here on 2026-08-03 and
    // REMOVED: it is pure cost. Rationale for the attempt was sound (the first
    // pass runs where only 8 rows are open, while ~69/99 are open in the final
    // vector) and it worked mechanically — 168-176 passes vs 8, 30/22 certified
    // contradictions, ~800 ms. But the published result was IDENTICAL to the
    // digit on both measured rows (prop_idx_7500 min -7.099 verified 30/99;
    // prop_idx_3343 min -10.386 verified 22/99, at 300 s AND 100 s).
    //
    // The contradiction count equals the verified count, which is the finding:
    // this lane proves the rows the ordinary phases already prove. Its reach
    // overlaps existing capability rather than extending past it, so paying it
    // twice buys nothing. Do not re-add without evidence that the lane can
    // close a row no other phase closes.

    // Root-only output-conditioned hidden treatment (exact dark gate
    // `NY_ROOT_OUTPUT_CONDITIONED_HEAD=1`). Admission requires one ordinary
    // unresolved OR row and an exact final Linear->ReLU->Linear head. The
    // two-seed backward conditions one crossing pre-activation coordinate on
    // that row's violation premise; the resulting box exists only in a private
    // map consumed by one ordinary same-row SpecCrownRequest replay.
    //
    // Publication is terminal-only: the helper rechecks that every other row
    // has an ordinary strict proof and returns only when the replay itself has
    // `LB > threshold`. The boolean receipt augments `verified_count` below;
    // ordinary numeric bounds remain untouched. No conditioned interval,
    // alpha, linear cache, or numeric marker can reach the root domain or BaB.
    // Every refusal/late result returns `None` without borrowing caller state
    // mutably.
    let output_conditioned_root_accepted = try_output_conditioned_root_refutation(
        verifier.config.verification_artifact_authority,
        graph,
        input,
        objectives,
        thresholds,
        &initial_obj_bounds,
        conjunctive,
        verifier.config.crown_backward_layers,
        &bootstrap.initial_node_bounds,
        bootstrap.root_alpha_state.as_ref(),
        engine,
        deadline,
    );
    if let Some(accepted) = output_conditioned_root_accepted.as_ref() {
        let row = accepted.row_index;
        info!(
            status = "accepted",
            row,
            target = %accepted.receipt.scope.target_preactivation,
            coordinate = accepted.target_coordinate,
            gamma_lower = accepted.gamma_lower(),
            gamma_upper = accepted.gamma_upper(),
            conditional_lower = f32::from_bits(accepted.receipt.conditional_lower_bits),
            threshold = f32::from_bits(accepted.receipt.threshold_bits),
            cache_published = false,
            "Output-conditioned hidden root treatment"
        );
    }

    let ordinary_verified_count = log_root_objective_bounds(&initial_obj_bounds, thresholds);
    let verified_count = output_conditioned_root_accepted
        .as_ref()
        .and_then(|accepted| {
            accepted.terminal_verified_count(
                ordinary_verified_count,
                initial_obj_bounds.len(),
                conjunctive,
            )
        })
        .unwrap_or(ordinary_verified_count);
    // DIAGNOSTIC (NY_LPOPT_DUMP): also record NY's ROOT alpha-CROWN per-objective
    // (per-margin) lower/upper bounds + thresholds to `<path>.margins`, so the
    // off-line LP (p*_LP) can be compared to NY's own root bound WITHOUT needing
    // `-v` info tracing. One line per objective: `idx lower upper threshold`.
    if let Ok(path) = std::env::var("NY_LPOPT_DUMP") {
        let mpath = format!("{path}.margins");
        let mut s = String::new();
        for (idx, ((lo, up), th)) in initial_obj_bounds.iter().zip(thresholds.iter()).enumerate() {
            s.push_str(&format!("{idx} {lo} {up} {th}\n"));
        }
        match std::fs::write(&mpath, s) {
            Ok(()) => eprintln!(
                "[lpopt-dump] wrote {} root margins to {mpath}",
                initial_obj_bounds.len()
            ),
            Err(e) => eprintln!("[lpopt-dump] margins write error on {mpath}: {e}"),
        }
    }
    // DIAGNOSTIC (NY_ROOT_WIDTH_PROBE=1): per-layer bound-width profile + output-margin
    // looseness decomposition at the ROOT domain (no split). Read-only / print-only;
    // never mutates the bootstrap or feeds any verdict. See diag/cifar100-root-width.
    if std::env::var("NY_ROOT_WIDTH_PROBE").ok().as_deref() == Some("1") {
        run_root_width_probe(
            graph,
            input,
            objectives,
            thresholds,
            engine,
            &bootstrap,
            &initial_obj_bounds,
        );
    }
    // DIAGNOSTIC (NY_ROOT_CROWN_INTERM_PROBE=1): per-ReLU pre-activation total box
    // width, computed TWO ways at the ROOT (no BaB split): (a) NY's frozen
    // forward-linear reference bound (what every BaB subdomain inherits) vs (b) a
    // sound CROWN BACKWARD from that pre-activation node to the input eps-box
    // (heuristic α), and optionally (c) the same CROWN backward with the warmup's
    // OPTIMIZED α. Decides whether the frozen intermediate bounds have real CROWN
    // headroom or are already CROWN-tight. Read-only / print-only; never mutates
    // the bootstrap or any verdict. Default-OFF ⇒ byte-identical.
    if std::env::var("NY_ROOT_CROWN_INTERM_PROBE").ok().as_deref() == Some("1") {
        run_root_crown_interm_probe(graph, input, engine, &bootstrap);
    }
    if let Some(result) = maybe_finish_at_root(
        lifecycle,
        initial_output.clone(),
        &initial_obj_bounds,
        thresholds,
        conjunctive,
        verified_count,
    ) {
        return Ok(MultiObjectiveRootOutcome::Finished(Box::new(result)));
    }

    // Use bab_timeout so post-BaB PGD reservation is respected (#4095).
    if lifecycle.start_time.elapsed() > bab_timeout {
        return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
            lifecycle.build_result_with_bounds(BabVerificationStatus::Timeout, initial_output),
        )));
    }

    // Final ownership handoff: the ordinary path moves each finalized tensor
    // directly into shared storage. A prebuilt critical-alpha setup already
    // owns the exact bounds used to construct its paired state, so discard the
    // now-redundant unshared map and retain that setup verbatim.
    let finalized_node_bounds = bootstrap.initial_node_bounds;
    let graph_setup = match prebuilt_graph_setup {
        Some(graph_setup) => {
            drop(finalized_node_bounds);
            graph_setup
        }
        None => build_graph_bab_setup_owned(graph, finalized_node_bounds),
    };
    let cut_pool = build_graph_cut_pool(
        graph,
        &graph_setup.initial_node_bounds_arc,
        &graph_setup.relu_nodes,
        &verifier.config,
    )?;

    // Clone initial_obj_bounds before moving into root domain — needed by
    // per-disjunct alpha optimization below to identify unverified disjuncts.
    let initial_obj_bounds_ref = initial_obj_bounds.clone();
    let root_boxes_tightened = root_intermediate_bounds_changed;
    let mut root_domain =
        MultiObjectiveGraphBabDomain::root_with_shared_node_bounds_and_aggregation(
            // HashMap keys and Arc handles are cloned; tensor buffers are shared.
            graph_setup.initial_node_bounds_arc.clone(),
            initial_obj_bounds,
            input,
            thresholds,
            false,
            ObjectiveAggregation::from_conjunctive(conjunctive),
        )?;
    debug_assert_eq!(
        root_domain.node_bounds().len(),
        graph_setup.initial_node_bounds_arc.len()
    );
    debug_assert!(
        graph_setup
            .initial_node_bounds_arc
            .iter()
            .all(|(name, setup_bounds)| root_domain
                .node_bounds()
                .get(name)
                .is_some_and(|domain_bounds| std::sync::Arc::ptr_eq(setup_bounds, domain_bounds))),
        "root domain must share finalized setup tensor buffers"
    );
    let root_alpha_was_prebuilt = prebuilt_root_alpha_state.is_some();
    root_domain.alpha_state = prebuilt_root_alpha_state.unwrap_or_else(|| {
        let root_alpha = root_bab_alpha_warm_start(
            bootstrap.root_alpha_state.as_ref(),
            root_boxes_tightened,
            fresh_atomic_root_alpha_installed,
            verifier.config.beta_iterations > 0,
        );
        build_root_alpha_state(
            graph,
            input,
            &root_domain.history,
            &graph_setup.initial_node_bounds_arc,
            root_alpha,
            verifier.config.beta_iterations > 0,
        )
    });
    if verifier.config.enable_clip_interm_domain {
        verifier.complete_clip_root_bounds_cache.store_finalized(
            graph,
            input,
            &graph_setup.initial_node_bounds_arc,
        );
    }

    // Per-disjunct alpha optimization (#4355): when enabled and the property
    // is disjunctive, optimize alpha independently for each unverified disjunct.
    // Merely enabling/attempting critical alpha cannot suppress this established
    // path: only an authoritative installed direct-C bound/state pair does.
    if critical_gpu_alpha_preserves_per_disjunct(critical_gpu_alpha_transport_tag.as_ref())
        && active_set_gpu_alpha_transport_tag.is_none()
        && active_set_scalar_cascade_transport_tag.is_none()
    {
        if let Some(root_alpha) = verifier
            .config
            .optimize_disjuncts_separately
            .then_some(bootstrap.root_alpha_state.as_ref())
            .flatten()
            .filter(|_| !conjunctive && objectives.len() > 1)
        {
            let per_disjunct = build_per_disjunct_alphas(
                graph,
                input,
                root_alpha,
                &bootstrap.alpha_config,
                bab_deadline,
                objectives,
                thresholds,
                &initial_obj_bounds_ref,
                &root_domain.history,
                &graph_setup.initial_node_bounds_arc,
                engine,
            )?;
            root_domain.set_per_disjunct_alphas(per_disjunct);
        }
    }

    attach_root_spec_cache(
        &mut root_domain,
        root_spec_cache,
        &root_spec_cache_active_indices,
        objectives.len(),
    );
    if let Some(tag) = critical_gpu_alpha_transport_tag {
        // The legacy cache was captured for a different alpha state. Only the
        // paired-best selected direct-C scalar/state is authoritative, so remove
        // that row's stale lA before children can consume it.
        root_domain.clear_cached_la_for_objective(tag.row_index);
        debug_assert_eq!(
            alpha_state_identity(root_domain.alpha_state()),
            Some(tag.state_identity),
            "published critical alpha state changed before root-domain install"
        );
    }
    if let Some(tag) = active_set_gpu_alpha_transport_tag {
        // A K-row shared alpha state invalidates every objective-local lA.
        // Per-disjunct alphas were suppressed above only after authoritative
        // active-set publication; refusals leave that ordinary fallback intact.
        invalidate_active_set_stale_transport(&mut root_domain);
        debug_assert!(root_domain.cached_las().iter().all(Option::is_none));
        debug_assert!(root_domain.per_disjunct_alphas().is_none());
        debug_assert_eq!(
            active_set_full_state_identity(root_domain.alpha_state()),
            Some(tag.state_identity),
            "published active-set alpha state changed before root-domain install"
        );
    }
    if let Some(tag) = active_set_scalar_cascade_transport_tag {
        // The installed scalar state differs from the state that certified the
        // retained active rows. Those retained bounds remain sound independent
        // certificates, but every state-dependent cache must be discarded.
        invalidate_active_set_stale_transport(&mut root_domain);
        debug_assert!(root_domain.cached_las().iter().all(Option::is_none));
        debug_assert!(root_domain.per_disjunct_alphas().is_none());
        debug_assert_eq!(
            alpha_state_identity(root_domain.alpha_state()),
            Some(tag.final_state_identity),
            "published cascade scalar state changed before root-domain install"
        );
        debug_assert_eq!(
            objective_bounds_fingerprint(root_domain.objective_bounds()),
            Some(tag.published_bounds_fingerprint),
            "published cascade certificate intersection changed before root-domain install"
        );
    }

    let batch_size = verifier.config.batch_size.max(1);
    let use_batched_gpu = engine.is_some() && batch_size > 1 && !conjunctive;
    let selective_gate = SelectiveRootAlphaGate::from_env();
    let fresh_atomic_root_alpha_was_published =
        fresh_atomic_root_alpha_installed && verifier.config.beta_iterations > 0;
    let selective_root_alpha_candidate = if !selective_gate.is_enabled() {
        None
    } else if !root_boxes_tightened {
        info!(
            status = "refused",
            reason = "root_boxes_unchanged",
            "Selective root alpha paired transport"
        );
        None
    } else if root_alpha_was_prebuilt || fresh_atomic_root_alpha_was_published {
        info!(
            status = "refused",
            reason = "authoritative_root_state_already_published",
            "Selective root alpha paired transport"
        );
        None
    } else if !use_batched_gpu {
        info!(
            status = "refused",
            reason = "shared_executor_unavailable",
            "Selective root alpha paired transport"
        );
        None
    } else if verifier.config.enable_cuts {
        info!(
            status = "refused",
            reason = "cuts_configured_require_established_fallback",
            "Selective root alpha paired transport"
        );
        None
    } else if !engine.is_some_and(|engine| {
        // Capability discovery can initialize a process-global accelerator.
        // Keep it behind exact opt-in and every cheaper eligibility refusal.
        crate::network::resnet_beta_gpu_enabled()
            && crate::network::resnet_beta_gpu_batched_enabled()
            && graph
                .nodes
                .values()
                .any(|node| matches!(node.layer, crate::Layer::Conv2d(_)))
            && sound_shared_gpu_available(engine, Some(bab_deadline))
    }) {
        info!(
            status = "refused",
            reason = "sound_shared_gpu_capability_unavailable",
            "Selective root alpha paired transport"
        );
        None
    } else if root_domain.per_disjunct_alphas().is_some() {
        info!(
            status = "refused",
            reason = "per_disjunct_alpha_active",
            "Selective root alpha paired transport"
        );
        None
    } else if let Some(root_alpha) = bootstrap.root_alpha_state.as_ref() {
        let candidate = GraphDomainAlphaState::try_from_root_alpha_state_borrowed_expanded(
            root_alpha,
            graph,
            &graph_setup.initial_node_bounds_arc,
            &root_domain.history,
            input,
        );
        if let Some(candidate) = candidate {
            info!(
                status = "armed",
                candidate_neurons = candidate.len(),
                "Selective root alpha paired transport"
            );
            Some(candidate)
        } else {
            info!(
                status = "refused",
                reason = "malformed_expanded_alpha_metadata",
                "Selective root alpha paired transport"
            );
            None
        }
    } else {
        info!(
            status = "refused",
            reason = "warmup_state_unavailable",
            "Selective root alpha paired transport"
        );
        None
    };
    let mode_str = if conjunctive {
        "conjunctive"
    } else {
        "disjunctive"
    };
    info!(
        "Multi-objective BaB ({}): {} objectives, {} ReLU nodes, {} cuts, batch_size={}, gpu_batched={}, timeout {:?}",
        mode_str,
        objectives.len(),
        graph_setup.relu_nodes.len(),
        cut_pool.len(),
        batch_size,
        use_batched_gpu,
        verifier.config.timeout
    );
    if crate::phase_telemetry::phase_telemetry_enabled() {
        let now = std::time::Instant::now();
        let remaining = bab_deadline.saturating_duration_since(now).as_secs_f64();
        crate::phase_telemetry::phase_marker(&format!(
            "multiobj-bab-ready root-elapsed={:.3}s remaining={remaining:.3}s",
            lifecycle.start_time.elapsed().as_secs_f64(),
        ));
    }

    Ok(MultiObjectiveRootOutcome::Continue(Box::new(
        MultiObjectiveRootState {
            initial_output,
            root_domain,
            selective_root_alpha_candidate,
            relu_nodes: graph_setup.relu_nodes,
            cut_pool,
            use_batched_gpu,
        },
    )))
}

/// Run the adaptive scalar/DAG coefficient backward, bypassing bounds-only
/// forward-linear/GPU root candidates while retaining its child-domain cache.
#[allow(clippy::too_many_arguments)]
fn run_adaptive_root_spec_candidate(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &ndarray::Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    alpha_state: Option<&crate::bounds::GraphAlphaState>,
    deadline: Option<std::time::Instant>,
    crown_backward_layers: Option<usize>,
) -> Result<(BoundedTensor, Option<CachedLinearBounds>)> {
    SpecCrownRequest::new(graph, input, spec_matrix, engine)
        .node_bounds(node_bounds)
        .alpha_state_opt(alpha_state)
        .deadline_opt(deadline)
        .truncate_after_opt(crown_backward_layers)
        .run_with_backward_cache()
}

/// Intersect two certified enclosures row by row. A disjoint element retains
/// `primary` while other overlapping rows still improve (never-worse fallback).
fn best_of_sound_root_spec_bounds(
    primary: BoundedTensor,
    secondary: &BoundedTensor,
) -> BoundedTensor {
    if primary.shape() != secondary.shape() {
        return primary;
    }
    let mut lower = primary.lower().clone();
    let mut upper = primary.upper().clone();
    ndarray::Zip::from(&mut lower)
        .and(&mut upper)
        .and(secondary.lower())
        .and(secondary.upper())
        .for_each(
            |primary_lower, primary_upper, &secondary_lower, &secondary_upper| {
                let intersect_lower = (*primary_lower).max(secondary_lower);
                let intersect_upper = (*primary_upper).min(secondary_upper);
                if intersect_lower <= intersect_upper {
                    *primary_lower = intersect_lower;
                    *primary_upper = intersect_upper;
                }
            },
        );
    BoundedTensor::new_allow_infinite(lower, upper).unwrap_or(primary)
}

const ROOT_CRITICAL_GPU_SPEC_MAX_RUNTIME: std::time::Duration = std::time::Duration::from_secs(1);
// Cooperative slice: boundary checks refuse late work/publication, while an
// already-launched device kernel necessarily runs to its backend completion.
const ROOT_CRITICAL_GPU_ALPHA_MAX_RUNTIME: std::time::Duration = std::time::Duration::from_secs(2);
const ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE: std::time::Duration = std::time::Duration::from_secs(12);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CriticalGpuSpecRefusal {
    DenseHeadStageNotSelected,
    ConjunctiveProperty,
    TruncatedBackward,
    InvalidInput,
    NoUnresolvedRow,
    TooManyUnresolvedRows,
    InsufficientHeadroom,
    NoSoundGpuRoute,
    GpuRouteUnavailable,
    GpuRouteError,
    CompletedAfterDeadline,
    InvalidCandidate,
    CandidateDisjoint,
}

impl CriticalGpuSpecRefusal {
    fn telemetry_reason(self) -> &'static str {
        match self {
            Self::DenseHeadStageNotSelected => "dense_head_stage_not_selected",
            Self::ConjunctiveProperty => "conjunctive_property",
            Self::TruncatedBackward => "truncated_backward",
            Self::InvalidInput => "invalid_input",
            Self::NoUnresolvedRow => "no_unresolved_row",
            Self::TooManyUnresolvedRows => "too_many_unresolved_rows",
            Self::InsufficientHeadroom => "insufficient_headroom",
            Self::NoSoundGpuRoute => "no_sound_gpu_route",
            Self::GpuRouteUnavailable => "gpu_route_unavailable",
            Self::GpuRouteError => "gpu_route_error",
            Self::CompletedAfterDeadline => "completed_after_deadline",
            Self::InvalidCandidate => "invalid_candidate",
            Self::CandidateDisjoint => "candidate_disjoint",
        }
    }
}

#[derive(Debug)]
struct CriticalGpuSpecPlan {
    row_index: usize,
    spec_matrix: ndarray::Array2<f32>,
    deadline: std::time::Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CriticalGpuSpecBackend {
    Local,
    Factory,
}

impl CriticalGpuSpecBackend {
    fn telemetry_name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Factory => "factory",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CriticalGpuSpecAccepted {
    historical_lower: f32,
    candidate_lower: f32,
    merged_lower: f32,
}

struct CriticalGpuSpecRoutedAttempt {
    backend: CriticalGpuSpecBackend,
    result: std::result::Result<CriticalGpuSpecAccepted, CriticalGpuSpecRefusal>,
}

#[derive(Clone, Copy)]
enum CriticalGpuSpecTelemetry<'a> {
    PlanRefused {
        reason: CriticalGpuSpecRefusal,
    },
    BackendSelected {
        backend: CriticalGpuSpecBackend,
        row: usize,
    },
    CandidateRefused {
        backend: CriticalGpuSpecBackend,
        row: usize,
        reason: CriticalGpuSpecRefusal,
    },
    Accepted {
        backend: CriticalGpuSpecBackend,
        row: usize,
        accepted: &'a CriticalGpuSpecAccepted,
    },
}

/// Pure formatting core for the critical-row candidate's print-only
/// diagnostics. The scorecard runner forces `RUST_LOG=error`, so these markers
/// intentionally bypass tracing, but only under the existing exact
/// `NY_PHASE_TELEMETRY=1` gate.
fn critical_gpu_spec_telemetry_line_if(
    enabled: bool,
    event: CriticalGpuSpecTelemetry<'_>,
) -> Option<String> {
    if !enabled {
        return None;
    }
    Some(match event {
        CriticalGpuSpecTelemetry::PlanRefused { reason } => format!(
            "[critical-gpu-spec] status=refused backend=unselected row=none reason={}",
            reason.telemetry_reason()
        ),
        CriticalGpuSpecTelemetry::BackendSelected { backend, row } => format!(
            "[critical-gpu-spec] status=selected backend={} row={row}",
            backend.telemetry_name()
        ),
        CriticalGpuSpecTelemetry::CandidateRefused {
            backend,
            row,
            reason,
        } => format!(
            "[critical-gpu-spec] status=refused backend={} row={row} reason={}",
            backend.telemetry_name(),
            reason.telemetry_reason()
        ),
        CriticalGpuSpecTelemetry::Accepted {
            backend,
            row,
            accepted,
        } => format!(
            "[critical-gpu-spec] status=accepted backend={} row={row} \
             historical_lower={} candidate_lower={} merged_lower={} lift={}",
            backend.telemetry_name(),
            accepted.historical_lower,
            accepted.candidate_lower,
            accepted.merged_lower,
            accepted.merged_lower - accepted.historical_lower
        ),
    })
}

fn emit_critical_gpu_spec_telemetry(event: CriticalGpuSpecTelemetry<'_>) {
    if !crate::phase_telemetry::phase_telemetry_enabled() {
        return;
    }
    if let Some(line) = critical_gpu_spec_telemetry_line_if(true, event) {
        eprintln!("{line}");
    }
}

fn root_critical_gpu_spec_enabled_from_value(enable: Option<&str>) -> bool {
    matches!(enable, Some("1"))
}

fn root_critical_gpu_spec_enabled() -> bool {
    root_critical_gpu_spec_enabled_from_value(
        std::env::var("NY_ROOT_CRITICAL_GPU_SPEC").ok().as_deref(),
    )
}

fn root_critical_gpu_alpha_enabled_from_value(enable: Option<&str>) -> bool {
    matches!(enable, Some("1"))
}

fn root_critical_gpu_alpha_enabled() -> bool {
    root_critical_gpu_alpha_enabled_from_value(
        std::env::var("NY_ROOT_CRITICAL_GPU_ALPHA").ok().as_deref(),
    )
}

fn root_critical_gpu_alpha_active_set_enabled_from_value(enable: Option<&str>) -> bool {
    matches!(enable, Some("1"))
}

/// The reader closure makes the nesting observable in tests: a closed bracket
/// lane must not even inspect the active-set environment variable.
fn root_critical_gpu_alpha_active_set_enabled_if<F>(bracket_enabled: bool, read: F) -> bool
where
    F: FnOnce() -> Option<String>,
{
    bracket_enabled && root_critical_gpu_alpha_active_set_enabled_from_value(read().as_deref())
}

fn root_critical_gpu_alpha_active_set_enabled_if_bracket(bracket_enabled: bool) -> bool {
    root_critical_gpu_alpha_active_set_enabled_if(bracket_enabled, || {
        std::env::var("NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET").ok()
    })
}

fn root_critical_gpu_alpha_active_set_cascade_enabled_from_value(enable: Option<&str>) -> bool {
    matches!(enable, Some("1"))
}

/// The reader seam makes the authority nesting testable. The cascade variable
/// is invisible unless the parent active-set lane has already been admitted.
fn root_critical_gpu_alpha_active_set_cascade_enabled_if<F>(
    active_set_enabled: bool,
    read: F,
) -> bool
where
    F: FnOnce() -> Option<String>,
{
    active_set_enabled
        && root_critical_gpu_alpha_active_set_cascade_enabled_from_value(read().as_deref())
}

fn root_critical_gpu_alpha_active_set_cascade_enabled_if_active(active_set_enabled: bool) -> bool {
    root_critical_gpu_alpha_active_set_cascade_enabled_if(active_set_enabled, || {
        std::env::var("NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET_CASCADE").ok()
    })
}

fn with_critical_gpu_alpha_gate<T>(enabled: bool, run: impl FnOnce() -> T) -> Option<T> {
    if enabled {
        Some(run())
    } else {
        None
    }
}

fn dispatch_critical_gpu_alpha_lr_bracket<T>(
    bracket_base_lr: Option<f32>,
    run_sealed_one_step: impl FnOnce() -> T,
    run_bracket: impl FnOnce(f32) -> T,
) -> T {
    match bracket_base_lr {
        Some(base_lr) => run_bracket(base_lr),
        None => run_sealed_one_step(),
    }
}

fn root_critical_gpu_deadline_with_runtime(
    now: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
    max_runtime: std::time::Duration,
) -> Option<std::time::Instant> {
    let private_deadline = now.checked_add(max_runtime)?;
    match authority_deadline {
        Some(authority)
            if authority.saturating_duration_since(now)
                >= max_runtime + ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE =>
        {
            Some(private_deadline)
        }
        Some(_) => None,
        None => Some(private_deadline),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_critical_gpu_spec_plan(
    dense_head_stage_selected: bool,
    conjunctive: bool,
    crown_backward_layers: Option<usize>,
    historical: &[(f32, f32)],
    full_spec_matrix: &ndarray::Array2<f32>,
    thresholds: &[f32],
    now: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
) -> std::result::Result<CriticalGpuSpecPlan, CriticalGpuSpecRefusal> {
    build_critical_gpu_spec_plan_with_runtime(
        dense_head_stage_selected,
        conjunctive,
        crown_backward_layers,
        historical,
        full_spec_matrix,
        thresholds,
        now,
        authority_deadline,
        ROOT_CRITICAL_GPU_SPEC_MAX_RUNTIME,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_critical_gpu_spec_plan_with_runtime(
    dense_head_stage_selected: bool,
    conjunctive: bool,
    crown_backward_layers: Option<usize>,
    historical: &[(f32, f32)],
    full_spec_matrix: &ndarray::Array2<f32>,
    thresholds: &[f32],
    now: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
    max_runtime: std::time::Duration,
) -> std::result::Result<CriticalGpuSpecPlan, CriticalGpuSpecRefusal> {
    if !dense_head_stage_selected {
        return Err(CriticalGpuSpecRefusal::DenseHeadStageNotSelected);
    }
    if conjunctive {
        return Err(CriticalGpuSpecRefusal::ConjunctiveProperty);
    }
    if crown_backward_layers.is_some() {
        return Err(CriticalGpuSpecRefusal::TruncatedBackward);
    }

    let row_count = thresholds.len();
    if row_count == 0
        || historical.len() != row_count
        || full_spec_matrix.nrows() != row_count
        || full_spec_matrix.ncols() == 0
        || historical
            .iter()
            .any(|&(lower, upper)| !root_interval_is_finite_ordered(lower, upper))
        || thresholds.iter().any(|value| !value.is_finite())
        || full_spec_matrix.iter().any(|value| !value.is_finite())
    {
        return Err(CriticalGpuSpecRefusal::InvalidInput);
    }

    let unresolved: Vec<_> = historical
        .iter()
        .copied()
        .zip(thresholds.iter().copied())
        .enumerate()
        .filter_map(|(row, ((lower, upper), threshold))| {
            (!root_prebound_certifies(lower, upper, threshold)).then_some(row)
        })
        .collect();
    let row_index = match unresolved.as_slice() {
        [] => return Err(CriticalGpuSpecRefusal::NoUnresolvedRow),
        [row] => *row,
        _ => return Err(CriticalGpuSpecRefusal::TooManyUnresolvedRows),
    };
    let deadline = root_critical_gpu_deadline_with_runtime(now, authority_deadline, max_runtime)
        .ok_or(CriticalGpuSpecRefusal::InsufficientHeadroom)?;
    let mut spec_matrix = ndarray::Array2::zeros((1, full_spec_matrix.ncols()));
    spec_matrix
        .row_mut(0)
        .assign(&full_spec_matrix.row(row_index));
    Ok(CriticalGpuSpecPlan {
        row_index,
        spec_matrix,
        deadline,
    })
}

fn root_critical_gpu_engine_supports_route(engine: Option<&dyn GemmEngine>) -> bool {
    engine
        .and_then(|candidate| candidate.as_gpu_crown_backward())
        .filter(|gpu| gpu.provides_sound_gpu_crown())
        .is_some_and(|gpu| gpu.provides_deadline_bounded_single_row_resnet_sound())
}

fn select_critical_gpu_spec_backend(engine: Option<&dyn GemmEngine>) -> CriticalGpuSpecBackend {
    if root_critical_gpu_engine_supports_route(engine) {
        CriticalGpuSpecBackend::Local
    } else {
        CriticalGpuSpecBackend::Factory
    }
}

fn root_critical_gpu_sound_route_available(
    engine: Option<&dyn GemmEngine>,
    deadline: std::time::Instant,
) -> bool {
    std::time::Instant::now() < deadline && root_critical_gpu_engine_supports_route(engine)
}

fn accept_critical_gpu_spec_candidate(
    plan: &CriticalGpuSpecPlan,
    historical: (f32, f32),
    candidate: &BoundedTensor,
    completed_at: std::time::Instant,
) -> std::result::Result<CriticalGpuSpecAccepted, CriticalGpuSpecRefusal> {
    if completed_at >= plan.deadline {
        return Err(CriticalGpuSpecRefusal::CompletedAfterDeadline);
    }
    if candidate.shape() != [1] || !root_interval_is_finite_ordered(historical.0, historical.1) {
        return Err(CriticalGpuSpecRefusal::InvalidCandidate);
    }
    let historical_lower = historical.0;
    let historical_upper = historical.1;
    let candidate_lower = candidate.lower()[[0]];
    let candidate_upper = candidate.upper()[[0]];
    if !root_interval_is_finite_ordered(candidate_lower, candidate_upper) {
        return Err(CriticalGpuSpecRefusal::InvalidCandidate);
    }
    if historical_lower.max(candidate_lower) > historical_upper.min(candidate_upper) {
        return Err(CriticalGpuSpecRefusal::CandidateDisjoint);
    }
    let merged_lower = historical_lower.max(candidate_lower);
    Ok(CriticalGpuSpecAccepted {
        historical_lower,
        candidate_lower,
        merged_lower,
    })
}

/// Publish only the accepted scalar lower bound. The candidate upper endpoint
/// is used above solely to prove overlap with the historical certified
/// interval; it must never replace the selected full-row result or its cache.
fn apply_critical_gpu_spec_lower_only(
    objective_bounds: &mut [(f32, f32)],
    row: usize,
    accepted: &CriticalGpuSpecAccepted,
    plan: &CriticalGpuSpecPlan,
) -> std::result::Result<(), CriticalGpuSpecRefusal> {
    apply_critical_gpu_spec_lower_only_with_clock(
        objective_bounds,
        row,
        accepted,
        plan,
        std::time::Instant::now,
    )
}

fn apply_critical_gpu_spec_lower_only_with_clock<F>(
    objective_bounds: &mut [(f32, f32)],
    row: usize,
    accepted: &CriticalGpuSpecAccepted,
    plan: &CriticalGpuSpecPlan,
    now: F,
) -> std::result::Result<(), CriticalGpuSpecRefusal>
where
    F: FnOnce() -> std::time::Instant,
{
    let Some((lower, _upper)) = objective_bounds.get_mut(row) else {
        return Err(CriticalGpuSpecRefusal::InvalidCandidate);
    };
    // This is the final authority check: sample after resolving the publication
    // target and immediately before the sole mutation. Factory unwinding,
    // result flattening, and telemetry setup may all consume the remaining
    // private budget after computation itself completed.
    if now() >= plan.deadline {
        return Err(CriticalGpuSpecRefusal::CompletedAfterDeadline);
    }
    *lower = accepted.merged_lower;
    Ok(())
}

fn run_critical_gpu_spec_candidate_on_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    engine: &dyn GemmEngine,
    plan: &CriticalGpuSpecPlan,
    historical: (f32, f32),
) -> std::result::Result<CriticalGpuSpecAccepted, CriticalGpuSpecRefusal> {
    // This checks the exact engine that executes the request. In particular,
    // the factory closure cannot admit a different engine and then accidentally
    // hand the WGPU caller's engine to SpecCrownRequest.
    if !root_critical_gpu_sound_route_available(Some(engine), plan.deadline) {
        return Err(CriticalGpuSpecRefusal::NoSoundGpuRoute);
    }
    let candidate = match SpecCrownRequest::new(graph, input, &plan.spec_matrix, Some(engine))
        .node_bounds(node_bounds)
        .deadline_opt(Some(plan.deadline))
        .run_fresh_slope_sound_gpu_bounds_only()
    {
        Ok(Some(candidate)) => candidate,
        Ok(None) => return Err(CriticalGpuSpecRefusal::GpuRouteUnavailable),
        Err(error) => {
            debug!(%error, "Critical-row fresh-slope GPU route errored");
            return Err(classify_critical_gpu_spec_error(&error));
        }
    };
    accept_critical_gpu_spec_candidate(plan, historical, &candidate, std::time::Instant::now())
}

type CriticalGpuSpecAttempt = std::result::Result<CriticalGpuSpecAccepted, CriticalGpuSpecRefusal>;

fn classify_critical_gpu_spec_error(error: &ny_core::NyError) -> CriticalGpuSpecRefusal {
    if error.is_deadline_exceeded() {
        CriticalGpuSpecRefusal::CompletedAfterDeadline
    } else {
        CriticalGpuSpecRefusal::GpuRouteError
    }
}

/// Flatten the factory accessor's `Result<Option<Result<_>>>` without allowing
/// any outer, absent, or inner failure to reach publication.
fn flatten_critical_gpu_spec_factory_result(
    nested: Result<Option<CriticalGpuSpecAttempt>>,
) -> CriticalGpuSpecAttempt {
    match nested {
        Ok(Some(inner)) => inner,
        Ok(None) => Err(CriticalGpuSpecRefusal::GpuRouteUnavailable),
        Err(error) => {
            debug!(%error, "Critical-row sound f64 factory route errored");
            Err(classify_critical_gpu_spec_error(&error))
        }
    }
}

fn run_critical_gpu_spec_candidate(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    engine: Option<&dyn GemmEngine>,
    plan: &CriticalGpuSpecPlan,
    historical: (f32, f32),
) -> CriticalGpuSpecRoutedAttempt {
    let backend = select_critical_gpu_spec_backend(engine);
    emit_critical_gpu_spec_telemetry(CriticalGpuSpecTelemetry::BackendSelected {
        backend,
        row: plan.row_index,
    });
    let result = match backend {
        CriticalGpuSpecBackend::Local => match engine {
            Some(local_engine) => run_critical_gpu_spec_candidate_on_engine(
                graph,
                input,
                node_bounds,
                local_engine,
                plan,
                historical,
            ),
            None => Err(CriticalGpuSpecRefusal::NoSoundGpuRoute),
        },
        CriticalGpuSpecBackend::Factory => flatten_critical_gpu_spec_factory_result(
            crate::sound_f64_gemm::with_engine_deadline(plan.deadline, |factory_engine| {
                // Capability check and the complete typed request both use this
                // exact factory engine inside the deadline-bearing closure.
                run_critical_gpu_spec_candidate_on_engine(
                    graph,
                    input,
                    node_bounds,
                    factory_engine,
                    plan,
                    historical,
                )
            }),
        ),
    };
    CriticalGpuSpecRoutedAttempt { backend, result }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveSetGpuAlphaRootRefusal {
    Plan(CriticalGpuSpecRefusal),
    Classification(ActiveSetGpuAlphaRefusal),
    Execution(ActiveSetGpuAlphaExecutionRefusal),
    FactoryUnavailable,
    FactoryError,
    CompletedAfterDeadline,
    HistoricalStateMismatch,
    CandidateDisjoint,
    NoCertifiedImprovement,
    SearchProvenanceMismatch,
}

impl ActiveSetGpuAlphaRootRefusal {
    fn telemetry_reason(self) -> &'static str {
        match self {
            Self::Plan(reason) => reason.telemetry_reason(),
            Self::Classification(reason) => reason.telemetry_reason(),
            Self::Execution(reason) => reason.telemetry_reason(),
            Self::FactoryUnavailable => "factory_unavailable",
            Self::FactoryError => "factory_error",
            Self::CompletedAfterDeadline => "completed_after_deadline",
            Self::HistoricalStateMismatch => "historical_state_mismatch",
            Self::CandidateDisjoint => "candidate_disjoint",
            Self::NoCertifiedImprovement => "no_certified_improvement",
            Self::SearchProvenanceMismatch => "search_provenance_mismatch",
        }
    }
}

impl From<ActiveSetGpuAlphaExecutionRefusal> for ActiveSetGpuAlphaRootRefusal {
    fn from(value: ActiveSetGpuAlphaExecutionRefusal) -> Self {
        Self::Execution(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ActiveSetGpuAlphaPublishedRow {
    source_row_index: usize,
    threshold: f32,
    historical_lower: f32,
    historical_upper: f32,
    candidate_lower: f32,
    merged_lower: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct ActiveSetGpuAlphaTransportTag {
    rows: Box<[ActiveSetGpuAlphaPublishedRow]>,
    score: ActiveSetGpuAlphaScore,
    state_identity: ActiveSetGpuAlphaFullStateIdentity,
    pair_fingerprint: ActiveSetCertifiedPairFingerprint,
    selected: ActiveSetGpuAlphaSelectedCandidate,
    base_lr: f32,
    evaluated_candidates: usize,
    gradient_replays: usize,
}

struct ActiveSetGpuAlphaRoutedAttempt {
    backend: CriticalGpuSpecBackend,
    result: std::result::Result<ActiveSetGpuAlphaExecutionOutput, ActiveSetGpuAlphaRootRefusal>,
}

fn classify_complete_active_set_gpu_alpha(
    historical: &[(f32, f32)],
    thresholds: &[f32],
) -> std::result::Result<ActiveSetGpuAlphaClassification, ActiveSetGpuAlphaRootRefusal> {
    if historical.is_empty() || historical.len() != thresholds.len() {
        return Err(ActiveSetGpuAlphaRootRefusal::Plan(
            CriticalGpuSpecRefusal::InvalidInput,
        ));
    }
    // Load-bearing strictness: equality stays unresolved. Do not validate row
    // payloads here; the classifier inspects K first so K>8 refuses before any
    // numerical/extraction/GPU work, even when a payload is malformed.
    let unresolved: Vec<_> = historical
        .iter()
        .copied()
        .zip(thresholds.iter().copied())
        .enumerate()
        .filter_map(|(source_row_index, ((lower, upper), threshold))| {
            (lower.partial_cmp(&threshold) != Some(std::cmp::Ordering::Greater)).then_some(
                ActiveSetUnresolvedRow::new(source_row_index, lower, upper, threshold),
            )
        })
        .collect();
    classify_active_set_gpu_alpha(&unresolved).map_err(ActiveSetGpuAlphaRootRefusal::Classification)
}

fn validate_active_set_root_surface(
    dense_head_stage_selected: bool,
    conjunctive: bool,
    crown_backward_layers: Option<usize>,
) -> std::result::Result<(), ActiveSetGpuAlphaRootRefusal> {
    if !dense_head_stage_selected {
        return Err(ActiveSetGpuAlphaRootRefusal::Plan(
            CriticalGpuSpecRefusal::DenseHeadStageNotSelected,
        ));
    }
    if conjunctive {
        return Err(ActiveSetGpuAlphaRootRefusal::Plan(
            CriticalGpuSpecRefusal::ConjunctiveProperty,
        ));
    }
    if crown_backward_layers.is_some() {
        return Err(ActiveSetGpuAlphaRootRefusal::Plan(
            CriticalGpuSpecRefusal::TruncatedBackward,
        ));
    }
    Ok(())
}

fn root_active_set_gpu_engine_supports_route(engine: Option<&dyn GemmEngine>, rows: usize) -> bool {
    engine
        .and_then(|candidate| candidate.as_gpu_crown_backward())
        .filter(|gpu| gpu.provides_sound_gpu_crown())
        .is_some_and(|gpu| {
            let capacity = gpu.deadline_bounded_resnet_sound_max_rows();
            capacity <= ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS && capacity >= rows
        })
}

fn select_active_set_gpu_backend(
    engine: Option<&dyn GemmEngine>,
    rows: usize,
) -> CriticalGpuSpecBackend {
    if root_active_set_gpu_engine_supports_route(engine, rows) {
        CriticalGpuSpecBackend::Local
    } else {
        CriticalGpuSpecBackend::Factory
    }
}

fn classify_active_set_gpu_alpha_factory_error(
    error: &ny_core::NyError,
) -> ActiveSetGpuAlphaRootRefusal {
    if error.is_deadline_exceeded() {
        ActiveSetGpuAlphaRootRefusal::Execution(CriticalGpuAlphaStepRefusal::DeadlineExpired.into())
    } else {
        ActiveSetGpuAlphaRootRefusal::FactoryError
    }
}

#[allow(clippy::too_many_arguments)]
fn run_active_set_gpu_alpha_candidate(
    plan: &mut ActiveSetGpuAlphaPlan,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    engine: Option<&dyn GemmEngine>,
    full_spec_matrix: &ndarray::Array2<f32>,
    hard_deadline: std::time::Instant,
    initial_state: &GraphDomainAlphaState,
    adaptive_config: &crate::beta_crown::config::AdaptiveOptConfig,
    base_lr: f32,
) -> ActiveSetGpuAlphaRoutedAttempt {
    let backend = select_active_set_gpu_backend(engine, plan.len());
    emit_active_set_gpu_alpha_telemetry(ActiveSetGpuAlphaTelemetry::BackendSelected {
        backend,
        rows: plan.len(),
    });
    let result = match backend {
        CriticalGpuSpecBackend::Local => match engine {
            Some(local_engine) => run_active_set_gpu_alpha_lr_bracket(
                plan,
                graph,
                input,
                node_bounds,
                local_engine,
                full_spec_matrix,
                hard_deadline,
                initial_state,
                adaptive_config,
                base_lr,
            )
            .map_err(ActiveSetGpuAlphaRootRefusal::Execution),
            None => Err(ActiveSetGpuAlphaRootRefusal::Execution(
                ActiveSetGpuAlphaExecutionRefusal::NoSoundGpuRoute,
            )),
        },
        CriticalGpuSpecBackend::Factory => {
            match crate::sound_f64_gemm::with_engine_deadline(hard_deadline, |factory_engine| {
                run_active_set_gpu_alpha_lr_bracket(
                    plan,
                    graph,
                    input,
                    node_bounds,
                    factory_engine,
                    full_spec_matrix,
                    hard_deadline,
                    initial_state,
                    adaptive_config,
                    base_lr,
                )
            }) {
                Ok(Some(inner)) => inner.map_err(ActiveSetGpuAlphaRootRefusal::Execution),
                Ok(None) => Err(ActiveSetGpuAlphaRootRefusal::FactoryUnavailable),
                Err(error) => {
                    debug!(%error, "Active-set alpha sound f64 factory route errored");
                    Err(classify_active_set_gpu_alpha_factory_error(&error))
                }
            }
        }
    };
    ActiveSetGpuAlphaRoutedAttempt { backend, result }
}

fn build_active_set_gpu_alpha_publication(
    objective_bounds: &[(f32, f32)],
    plan: &ActiveSetGpuAlphaPlan,
    output: ActiveSetGpuAlphaExecutionOutput,
    hard_deadline: std::time::Instant,
) -> std::result::Result<CriticalGpuAlphaRootPair, ActiveSetGpuAlphaRootRefusal> {
    build_active_set_gpu_alpha_publication_with_clock(
        objective_bounds,
        plan,
        output,
        hard_deadline,
        std::time::Instant::now,
    )
}

fn build_active_set_gpu_alpha_publication_with_clock<F>(
    objective_bounds: &[(f32, f32)],
    plan: &ActiveSetGpuAlphaPlan,
    output: ActiveSetGpuAlphaExecutionOutput,
    hard_deadline: std::time::Instant,
    now: F,
) -> std::result::Result<CriticalGpuAlphaRootPair, ActiveSetGpuAlphaRootRefusal>
where
    F: FnOnce() -> std::time::Instant,
{
    output
        .validate(plan)
        .map_err(ActiveSetGpuAlphaRootRefusal::Execution)?;
    let selected_pair = output.selected_pair();
    let candidate_lower: Vec<f32> = selected_pair.bounds().lower().iter().copied().collect();
    let candidate_upper: Vec<f32> = selected_pair.bounds().upper().iter().copied().collect();
    if candidate_lower.len() != plan.len() || candidate_upper.len() != plan.len() {
        return Err(ActiveSetGpuAlphaRootRefusal::SearchProvenanceMismatch);
    }

    let mut published_bounds = objective_bounds.to_vec();
    let mut published_rows = Vec::with_capacity(plan.len());
    let mut improved = false;
    for (active_ordinal, row) in plan.rows().iter().enumerate() {
        let source_row_index = row.source_row_index();
        let Some(historical) = objective_bounds.get(source_row_index).copied() else {
            return Err(ActiveSetGpuAlphaRootRefusal::HistoricalStateMismatch);
        };
        if historical.0.to_bits() != row.historical_lower().to_bits()
            || historical.1.to_bits() != row.historical_upper().to_bits()
            || !root_interval_is_finite_ordered(historical.0, historical.1)
        {
            return Err(ActiveSetGpuAlphaRootRefusal::HistoricalStateMismatch);
        }
        let lower = candidate_lower[active_ordinal];
        let upper = candidate_upper[active_ordinal];
        if !root_interval_is_finite_ordered(lower, upper) {
            return Err(ActiveSetGpuAlphaRootRefusal::SearchProvenanceMismatch);
        }
        if historical.0.max(lower) > historical.1.min(upper) {
            return Err(ActiveSetGpuAlphaRootRefusal::CandidateDisjoint);
        }
        let merged_lower = historical.0.max(lower);
        improved |= merged_lower > historical.0;
        published_bounds[source_row_index].0 = merged_lower;
        published_rows.push(ActiveSetGpuAlphaPublishedRow {
            source_row_index,
            threshold: row.threshold(),
            historical_lower: historical.0,
            historical_upper: historical.1,
            candidate_lower: lower,
            merged_lower,
        });
    }
    if !improved {
        return Err(ActiveSetGpuAlphaRootRefusal::NoCertifiedImprovement);
    }

    let tag = ActiveSetGpuAlphaTransportTag {
        rows: published_rows.into_boxed_slice(),
        score: selected_pair.score(),
        state_identity: selected_pair.state_identity(),
        pair_fingerprint: selected_pair.fingerprint(),
        selected: output.selected(),
        base_lr: output.base_lr(),
        evaluated_candidates: output.candidate_traces().len(),
        gradient_replays: output.gradient_replays(),
    };
    let (_selected_bounds, selected_state) = output.into_selected_pair().into_bound_state_pair();
    if active_set_full_state_identity(&selected_state) != Some(tag.state_identity) {
        return Err(ActiveSetGpuAlphaRootRefusal::SearchProvenanceMismatch);
    }
    // The single authority sample follows all validation/allocation and
    // immediately precedes publication of the complete vector/state pair.
    if now() >= hard_deadline {
        return Err(ActiveSetGpuAlphaRootRefusal::CompletedAfterDeadline);
    }
    Ok(CriticalGpuAlphaRootPair {
        objective_bounds: published_bounds,
        alpha_state: selected_state,
        transport_tag: None,
        active_set_transport_tag: Some(tag),
        active_set_scalar_cascade_transport_tag: None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveSetScalarCascadeRefusal {
    InvalidInput,
    NoSurvivor,
    TooManySurvivors { count: usize },
    DeadlineExpired,
    ActivePairMismatch,
    Scalar(CriticalGpuAlphaRefusal),
    CertifiedIntersectionMismatch,
    CompletedAfterDeadline,
}

impl ActiveSetScalarCascadeRefusal {
    fn telemetry_reason(self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
            Self::NoSurvivor => "no_survivor",
            Self::TooManySurvivors { .. } => "too_many_survivors",
            Self::DeadlineExpired => "deadline_expired",
            Self::ActivePairMismatch => "active_pair_mismatch",
            Self::Scalar(reason) => reason.telemetry_reason(),
            Self::CertifiedIntersectionMismatch => "certified_intersection_mismatch",
            Self::CompletedAfterDeadline => "completed_after_deadline",
        }
    }
}

/// Provenance for a two-certificate composition.
///
/// `active_set` binds the K-row direct-C enclosure to the state that produced
/// it. `CriticalGpuAlphaRootPair::transport_tag` separately binds the scalar
/// survivor enclosure to the final state installed on the root. The published
/// root bounds are the per-row intersection of those independently sound
/// enclosures: every non-survivor endpoint is retained bit-for-bit from the
/// active publication, while the survivor lower endpoint is the scalar
/// publication's `max(active_lower, scalar_lower)`. Bounds do not become
/// unsound when the heuristic alpha state changes; this tag makes that
/// cross-state intersection explicit instead of pretending every retained
/// active bound was evaluated under the final scalar state.
#[derive(Debug, Clone, PartialEq)]
struct ActiveSetScalarCascadeTransportTag {
    survivor_row: usize,
    active_set: ActiveSetGpuAlphaTransportTag,
    final_state_identity: CriticalGpuAlphaStateIdentity,
    published_bounds_fingerprint: u64,
}

const ACTIVE_SET_SCALAR_CASCADE_BOUNDS_DOMAIN: &[u8] =
    b"ny-active-set-scalar-cascade-certified-intersection-v1";

fn objective_bounds_fingerprint(bounds: &[(f32, f32)]) -> Option<u64> {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    fingerprint_bytes(&mut hash, ACTIVE_SET_SCALAR_CASCADE_BOUNDS_DOMAIN);
    fingerprint_u64(&mut hash, bounds.len() as u64);
    for &(lower, upper) in bounds {
        if !root_interval_is_finite_ordered(lower, upper) {
            return None;
        }
        fingerprint_u64(&mut hash, u64::from(lower.to_bits()));
        fingerprint_u64(&mut hash, u64::from(upper.to_bits()));
    }
    Some(hash)
}

fn classify_active_set_scalar_cascade_survivor(
    objective_bounds: &[(f32, f32)],
    thresholds: &[f32],
) -> std::result::Result<usize, ActiveSetScalarCascadeRefusal> {
    if objective_bounds.is_empty() || objective_bounds.len() != thresholds.len() {
        return Err(ActiveSetScalarCascadeRefusal::InvalidInput);
    }
    let mut survivor = None;
    let mut survivor_count = 0usize;
    for (row, (&(lower, upper), &threshold)) in objective_bounds.iter().zip(thresholds).enumerate()
    {
        if !root_interval_is_finite_ordered(lower, upper) || !threshold.is_finite() {
            return Err(ActiveSetScalarCascadeRefusal::InvalidInput);
        }
        if !root_prebound_certifies(lower, upper, threshold) {
            survivor_count += 1;
            survivor.get_or_insert(row);
        }
    }
    match (survivor_count, survivor) {
        (0, _) => Err(ActiveSetScalarCascadeRefusal::NoSurvivor),
        (1, Some(row)) => Ok(row),
        (count, _) => Err(ActiveSetScalarCascadeRefusal::TooManySurvivors { count }),
    }
}

fn build_active_set_scalar_cascade_plan(
    objective_bounds: &[(f32, f32)],
    full_spec_matrix: &ndarray::Array2<f32>,
    thresholds: &[f32],
    original_hard_deadline: std::time::Instant,
    now: std::time::Instant,
) -> std::result::Result<CriticalGpuSpecPlan, ActiveSetScalarCascadeRefusal> {
    let row_index = classify_active_set_scalar_cascade_survivor(objective_bounds, thresholds)?;
    if full_spec_matrix.nrows() != objective_bounds.len()
        || full_spec_matrix.ncols() == 0
        || full_spec_matrix.iter().any(|value| !value.is_finite())
    {
        return Err(ActiveSetScalarCascadeRefusal::InvalidInput);
    }
    // The cascade receives the active lane's original authority deadline
    // verbatim. It must never derive a new `now + runtime` slice.
    if now >= original_hard_deadline {
        return Err(ActiveSetScalarCascadeRefusal::DeadlineExpired);
    }
    let mut spec_matrix = ndarray::Array2::zeros((1, full_spec_matrix.ncols()));
    spec_matrix
        .row_mut(0)
        .assign(&full_spec_matrix.row(row_index));
    Ok(CriticalGpuSpecPlan {
        row_index,
        spec_matrix,
        deadline: original_hard_deadline,
    })
}

fn validate_active_set_pair_for_cascade<'a>(
    active_pair: &'a CriticalGpuAlphaRootPair,
    thresholds: &[f32],
) -> std::result::Result<&'a ActiveSetGpuAlphaTransportTag, ActiveSetScalarCascadeRefusal> {
    if active_pair.transport_tag.is_some()
        || active_pair
            .active_set_scalar_cascade_transport_tag
            .is_some()
        || active_pair.objective_bounds.len() != thresholds.len()
    {
        return Err(ActiveSetScalarCascadeRefusal::ActivePairMismatch);
    }
    let tag = active_pair
        .active_set_transport_tag
        .as_ref()
        .ok_or(ActiveSetScalarCascadeRefusal::ActivePairMismatch)?;
    if !(2..=ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&tag.rows.len())
        || active_set_full_state_identity(&active_pair.alpha_state) != Some(tag.state_identity)
    {
        return Err(ActiveSetScalarCascadeRefusal::ActivePairMismatch);
    }
    let mut seen = std::collections::BTreeSet::new();
    for row in tag.rows.iter() {
        let Some(&(published_lower, published_upper)) =
            active_pair.objective_bounds.get(row.source_row_index)
        else {
            return Err(ActiveSetScalarCascadeRefusal::ActivePairMismatch);
        };
        let Some(&threshold) = thresholds.get(row.source_row_index) else {
            return Err(ActiveSetScalarCascadeRefusal::ActivePairMismatch);
        };
        if !seen.insert(row.source_row_index)
            || threshold.to_bits() != row.threshold.to_bits()
            || !root_interval_is_finite_ordered(row.historical_lower, row.historical_upper)
            || !row.candidate_lower.is_finite()
            || row.merged_lower.to_bits() != row.historical_lower.max(row.candidate_lower).to_bits()
            || published_lower.to_bits() != row.merged_lower.to_bits()
            || published_upper.to_bits() != row.historical_upper.to_bits()
        {
            return Err(ActiveSetScalarCascadeRefusal::ActivePairMismatch);
        }
    }
    Ok(tag)
}

fn build_active_set_scalar_cascade_publication_with_clock<F>(
    active_pair: &CriticalGpuAlphaRootPair,
    scalar_publication: CriticalGpuAlphaPublication,
    plan: &CriticalGpuSpecPlan,
    thresholds: &[f32],
    original_hard_deadline: std::time::Instant,
    now: F,
) -> std::result::Result<CriticalGpuAlphaRootPair, ActiveSetScalarCascadeRefusal>
where
    F: FnOnce() -> std::time::Instant,
{
    let active_tag = validate_active_set_pair_for_cascade(active_pair, thresholds)?;
    let survivor_row =
        classify_active_set_scalar_cascade_survivor(&active_pair.objective_bounds, thresholds)?;
    if survivor_row != plan.row_index || plan.deadline != original_hard_deadline {
        return Err(ActiveSetScalarCascadeRefusal::CertifiedIntersectionMismatch);
    }

    let scalar_pair = scalar_publication.pair;
    if scalar_pair.active_set_transport_tag.is_some()
        || scalar_pair
            .active_set_scalar_cascade_transport_tag
            .is_some()
        || scalar_pair.objective_bounds.len() != active_pair.objective_bounds.len()
    {
        return Err(ActiveSetScalarCascadeRefusal::CertifiedIntersectionMismatch);
    }
    let scalar_tag = scalar_pair
        .transport_tag
        .as_ref()
        .ok_or(ActiveSetScalarCascadeRefusal::CertifiedIntersectionMismatch)?;
    if scalar_tag.row_index != survivor_row
        || scalar_tag.historical_lower.to_bits()
            != active_pair.objective_bounds[survivor_row].0.to_bits()
        || alpha_state_identity(&scalar_pair.alpha_state) != Some(scalar_tag.state_identity)
    {
        return Err(ActiveSetScalarCascadeRefusal::CertifiedIntersectionMismatch);
    }
    for (row, (&active, &published)) in active_pair
        .objective_bounds
        .iter()
        .zip(&scalar_pair.objective_bounds)
        .enumerate()
    {
        let expected_lower = if row == survivor_row {
            scalar_tag.merged_lower
        } else {
            active.0
        };
        if published.0.to_bits() != expected_lower.to_bits()
            || published.1.to_bits() != active.1.to_bits()
        {
            return Err(ActiveSetScalarCascadeRefusal::CertifiedIntersectionMismatch);
        }
    }
    let published_bounds_fingerprint = objective_bounds_fingerprint(&scalar_pair.objective_bounds)
        .ok_or(ActiveSetScalarCascadeRefusal::CertifiedIntersectionMismatch)?;
    let cascade_tag = ActiveSetScalarCascadeTransportTag {
        survivor_row,
        active_set: active_tag.clone(),
        final_state_identity: scalar_tag.state_identity,
        published_bounds_fingerprint,
    };
    // Final authority sample follows every validation, clone, allocation, and
    // fingerprint operation. A late scalar result cannot displace the accepted
    // active pair.
    if now() >= original_hard_deadline {
        return Err(ActiveSetScalarCascadeRefusal::CompletedAfterDeadline);
    }
    Ok(CriticalGpuAlphaRootPair {
        objective_bounds: scalar_pair.objective_bounds,
        alpha_state: scalar_pair.alpha_state,
        transport_tag: scalar_pair.transport_tag,
        active_set_transport_tag: None,
        active_set_scalar_cascade_transport_tag: Some(cascade_tag),
    })
}

fn retain_active_set_pair_on_cascade_refusal(
    active_pair: CriticalGpuAlphaRootPair,
    cascade: std::result::Result<CriticalGpuAlphaRootPair, ActiveSetScalarCascadeRefusal>,
) -> (
    CriticalGpuAlphaRootPair,
    Option<ActiveSetScalarCascadeRefusal>,
) {
    match cascade {
        Ok(pair) => (pair, None),
        Err(reason) => (active_pair, Some(reason)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CriticalGpuAlphaRefusal {
    Plan(CriticalGpuSpecRefusal),
    Step(CriticalGpuAlphaStepRefusal),
    FactoryUnavailable,
    FactoryError,
    CompletedAfterDeadline,
    InvalidInitialCandidate,
    InvalidFinalCandidate,
    CandidateDisjoint,
    NoCertifiedImprovement,
    StateIdentityMismatch,
    SearchProvenanceMismatch,
}

impl CriticalGpuAlphaRefusal {
    fn telemetry_reason(self) -> &'static str {
        match self {
            Self::Plan(reason) => reason.telemetry_reason(),
            Self::Step(reason) => reason.telemetry_reason(),
            Self::FactoryUnavailable => "factory_unavailable",
            Self::FactoryError => "factory_error",
            Self::CompletedAfterDeadline => "completed_after_deadline",
            Self::InvalidInitialCandidate => "invalid_initial_candidate",
            Self::InvalidFinalCandidate => "invalid_final_candidate",
            Self::CandidateDisjoint => "candidate_disjoint",
            Self::NoCertifiedImprovement => "no_certified_improvement",
            Self::StateIdentityMismatch => "state_identity_mismatch",
            Self::SearchProvenanceMismatch => "search_provenance_mismatch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CriticalGpuAlphaSelectedPair {
    Initial,
    Final,
}

impl CriticalGpuAlphaSelectedPair {
    fn telemetry_name(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Final => "final",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CriticalGpuAlphaBracketTag {
    selected_ordinal: usize,
    selected_lr: f32,
    evaluated_candidates: usize,
    gradient_replays: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CriticalGpuAlphaTransportTag {
    row_index: usize,
    historical_lower: f32,
    initial_lower: f32,
    final_lower: f32,
    merged_lower: f32,
    selected_pair: CriticalGpuAlphaSelectedPair,
    state_identity: CriticalGpuAlphaStateIdentity,
    bracket: Option<CriticalGpuAlphaBracketTag>,
}

fn critical_gpu_alpha_preserves_per_disjunct(
    installed: Option<&CriticalGpuAlphaTransportTag>,
) -> bool {
    installed.is_none()
}

fn invalidate_active_set_stale_transport(root: &mut MultiObjectiveGraphBabDomain) {
    let objective_count = root.objective_bounds().len();
    root.set_cached_las(vec![None; objective_count])
        .expect("root objective/cache lengths were constructed together");
    root.clear_per_disjunct_alphas();
}

#[derive(Debug)]
struct CriticalGpuAlphaRootPair {
    objective_bounds: Vec<(f32, f32)>,
    alpha_state: GraphDomainAlphaState,
    transport_tag: Option<CriticalGpuAlphaTransportTag>,
    active_set_transport_tag: Option<ActiveSetGpuAlphaTransportTag>,
    active_set_scalar_cascade_transport_tag: Option<ActiveSetScalarCascadeTransportTag>,
}

#[derive(Debug)]
struct CriticalGpuAlphaPublication {
    pair: CriticalGpuAlphaRootPair,
    accepted: CriticalGpuSpecAccepted,
}

struct CriticalGpuAlphaRootArtifacts {
    graph_setup: GraphBabSetup,
    pair: CriticalGpuAlphaRootPair,
}

struct CriticalGpuAlphaRoutedAttempt {
    backend: CriticalGpuSpecBackend,
    result: std::result::Result<CriticalGpuAlphaStepOutput, CriticalGpuAlphaRefusal>,
}

#[derive(Clone, Copy)]
enum CriticalGpuAlphaTelemetry<'a> {
    PlanRefused {
        reason: CriticalGpuAlphaRefusal,
    },
    BackendSelected {
        backend: CriticalGpuSpecBackend,
        row: usize,
    },
    CandidateRefused {
        backend: CriticalGpuSpecBackend,
        row: usize,
        reason: CriticalGpuAlphaRefusal,
    },
    BracketCandidate {
        row: usize,
        candidate: &'a CriticalGpuAlphaCandidateTrace,
    },
    Accepted {
        backend: CriticalGpuSpecBackend,
        tag: &'a CriticalGpuAlphaTransportTag,
    },
}

fn critical_gpu_alpha_telemetry_line_if(
    enabled: bool,
    event: CriticalGpuAlphaTelemetry<'_>,
) -> Option<String> {
    if !enabled {
        return None;
    }
    Some(match event {
        CriticalGpuAlphaTelemetry::PlanRefused { reason } => format!(
            "[critical-gpu-alpha] status=refused backend=unselected row=none reason={}",
            reason.telemetry_reason()
        ),
        CriticalGpuAlphaTelemetry::BackendSelected { backend, row } => format!(
            "[critical-gpu-alpha] status=selected backend={} row={row}",
            backend.telemetry_name()
        ),
        CriticalGpuAlphaTelemetry::CandidateRefused {
            backend,
            row,
            reason,
        } => format!(
            "[critical-gpu-alpha] status=refused backend={} row={row} reason={}",
            backend.telemetry_name(),
            reason.telemetry_reason()
        ),
        CriticalGpuAlphaTelemetry::BracketCandidate { row, candidate } => format!(
            "[critical-gpu-alpha-candidate] row={row} ordinal={} t={} lr={} lower={} lift={} \
             alpha_params={} fp={:016x}",
            candidate.ordinal,
            candidate.adam_t,
            candidate.alpha_lr,
            candidate.lower,
            candidate.lift_from_initial,
            candidate.state_identity.parameter_count,
            candidate.state_identity.fingerprint,
        ),
        CriticalGpuAlphaTelemetry::Accepted { backend, tag } => match tag.bracket {
            Some(bracket) => format!(
                "[critical-gpu-alpha] status=accepted backend={} row={} \
                 initial_lower={} final_lower={} selected={} historical_lower={} merged_lower={} lift={} \
                 selected_ordinal={} selected_lr={} candidates={} replays={} \
                 alpha_params={} alpha_fingerprint={:016x} cache=invalidated",
                backend.telemetry_name(),
                tag.row_index,
                tag.initial_lower,
                tag.final_lower,
                tag.selected_pair.telemetry_name(),
                tag.historical_lower,
                tag.merged_lower,
                tag.merged_lower - tag.historical_lower,
                bracket.selected_ordinal,
                bracket.selected_lr,
                bracket.evaluated_candidates,
                bracket.gradient_replays,
                tag.state_identity.parameter_count,
                tag.state_identity.fingerprint,
            ),
            None => format!(
                "[critical-gpu-alpha] status=accepted backend={} row={} \
                 initial_lower={} final_lower={} selected={} historical_lower={} merged_lower={} lift={} \
                 alpha_params={} alpha_fingerprint={:016x} cache=invalidated",
                backend.telemetry_name(),
                tag.row_index,
                tag.initial_lower,
                tag.final_lower,
                tag.selected_pair.telemetry_name(),
                tag.historical_lower,
                tag.merged_lower,
                tag.merged_lower - tag.historical_lower,
                tag.state_identity.parameter_count,
                tag.state_identity.fingerprint,
            ),
        },
    })
}

fn emit_critical_gpu_alpha_telemetry(event: CriticalGpuAlphaTelemetry<'_>) {
    if !crate::phase_telemetry::phase_telemetry_enabled() {
        return;
    }
    if let Some(line) = critical_gpu_alpha_telemetry_line_if(true, event) {
        eprintln!("{line}");
    }
}

#[derive(Clone, Copy)]
enum ActiveSetGpuAlphaTelemetry<'a> {
    PlanRefused {
        rows: Option<usize>,
        reason: ActiveSetGpuAlphaRootRefusal,
    },
    BackendSelected {
        backend: CriticalGpuSpecBackend,
        rows: usize,
    },
    Candidate {
        rows: usize,
        trace: &'a ActiveSetGpuAlphaCandidateTrace,
    },
    CandidateRefused {
        backend: CriticalGpuSpecBackend,
        rows: usize,
        reason: ActiveSetGpuAlphaRootRefusal,
    },
    Accepted {
        backend: CriticalGpuSpecBackend,
        tag: &'a ActiveSetGpuAlphaTransportTag,
    },
}

fn active_set_gpu_alpha_telemetry_line_if(
    enabled: bool,
    event: ActiveSetGpuAlphaTelemetry<'_>,
) -> Option<String> {
    if !enabled {
        return None;
    }
    Some(match event {
        ActiveSetGpuAlphaTelemetry::PlanRefused { rows, reason } => format!(
            "[active-set-gpu-alpha] status=refused backend=unselected rows={} reason={}",
            rows.map_or_else(|| "none".to_string(), |count| count.to_string()),
            reason.telemetry_reason(),
        ),
        ActiveSetGpuAlphaTelemetry::BackendSelected { backend, rows } => format!(
            "[active-set-gpu-alpha] status=selected backend={} rows={rows}",
            backend.telemetry_name(),
        ),
        ActiveSetGpuAlphaTelemetry::Candidate { rows, trace } => format!(
            "[active-set-gpu-alpha-candidate] rows={rows} ordinal={} lr={} \
             certified={} min_slack={} negative_slack_sum={} alpha_params={} \
             state_fp={:016x} pair_fp={:016x}",
            trace.ordinal(),
            trace.alpha_lr(),
            trace.score().rows_certified(),
            trace.score().min_slack(),
            trace.score().negative_slack_sum(),
            trace.state_identity().parameter_count(),
            trace.state_identity().fingerprint(),
            trace.pair_fingerprint().value(),
        ),
        ActiveSetGpuAlphaTelemetry::CandidateRefused {
            backend,
            rows,
            reason,
        } => format!(
            "[active-set-gpu-alpha] status=refused backend={} rows={rows} reason={}",
            backend.telemetry_name(),
            reason.telemetry_reason(),
        ),
        ActiveSetGpuAlphaTelemetry::Accepted { backend, tag } => {
            let source_rows = tag
                .rows
                .iter()
                .map(|row| row.source_row_index.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let selected = match tag.selected {
                ActiveSetGpuAlphaSelectedCandidate::Initial => "initial".to_string(),
                ActiveSetGpuAlphaSelectedCandidate::Candidate { ordinal } => {
                    format!("candidate-{ordinal}")
                }
            };
            format!(
                "[active-set-gpu-alpha] status=accepted backend={} rows={} \
                 source_rows={} selected={} base_lr={} candidates={} replays={} \
                 certified={} min_slack={} negative_slack_sum={} alpha_params={} \
                 state_fp={:016x} pair_fp={:016x} cache=all_invalidated",
                backend.telemetry_name(),
                tag.rows.len(),
                source_rows,
                selected,
                tag.base_lr,
                tag.evaluated_candidates,
                tag.gradient_replays,
                tag.score.rows_certified(),
                tag.score.min_slack(),
                tag.score.negative_slack_sum(),
                tag.state_identity.parameter_count(),
                tag.state_identity.fingerprint(),
                tag.pair_fingerprint.value(),
            )
        }
    })
}

fn emit_active_set_gpu_alpha_telemetry(event: ActiveSetGpuAlphaTelemetry<'_>) {
    if !crate::phase_telemetry::phase_telemetry_enabled() {
        return;
    }
    if let Some(line) = active_set_gpu_alpha_telemetry_line_if(true, event) {
        eprintln!("{line}");
    }
}

#[derive(Clone, Copy)]
enum ActiveSetScalarCascadeTelemetry<'a> {
    Refused {
        backend: Option<CriticalGpuSpecBackend>,
        row: Option<usize>,
        reason: ActiveSetScalarCascadeRefusal,
    },
    BackendSelected {
        backend: CriticalGpuSpecBackend,
        row: usize,
    },
    Candidate {
        row: usize,
        candidate: &'a CriticalGpuAlphaCandidateTrace,
    },
    Accepted {
        backend: CriticalGpuSpecBackend,
        tag: &'a ActiveSetScalarCascadeTransportTag,
        scalar: &'a CriticalGpuAlphaTransportTag,
    },
}

fn active_set_scalar_cascade_telemetry_line_if(
    enabled: bool,
    event: ActiveSetScalarCascadeTelemetry<'_>,
) -> Option<String> {
    if !enabled {
        return None;
    }
    Some(match event {
        ActiveSetScalarCascadeTelemetry::Refused {
            backend,
            row,
            reason,
        } => {
            let survivors = match reason {
                ActiveSetScalarCascadeRefusal::NoSurvivor => "0".to_string(),
                ActiveSetScalarCascadeRefusal::TooManySurvivors { count } => count.to_string(),
                _ => "unknown".to_string(),
            };
            format!(
                "[active-set-gpu-alpha-cascade] status=refused backend={} row={} \
                 survivors={survivors} reason={}",
                backend.map_or("unselected", CriticalGpuSpecBackend::telemetry_name),
                row.map_or_else(|| "none".to_string(), |value| value.to_string()),
                reason.telemetry_reason(),
            )
        }
        ActiveSetScalarCascadeTelemetry::BackendSelected { backend, row } => format!(
            "[active-set-gpu-alpha-cascade] status=selected backend={} row={row}",
            backend.telemetry_name(),
        ),
        ActiveSetScalarCascadeTelemetry::Candidate { row, candidate } => format!(
            "[active-set-gpu-alpha-cascade-candidate] row={row} ordinal={} t={} lr={} \
             lower={} lift={} alpha_params={} fp={:016x}",
            candidate.ordinal,
            candidate.adam_t,
            candidate.alpha_lr,
            candidate.lower,
            candidate.lift_from_initial,
            candidate.state_identity.parameter_count,
            candidate.state_identity.fingerprint,
        ),
        ActiveSetScalarCascadeTelemetry::Accepted {
            backend,
            tag,
            scalar,
        } => format!(
            "[active-set-gpu-alpha-cascade] status=accepted backend={} row={} \
             historical_lower={} merged_lower={} lift={} active_rows={} \
             active_state_fp={:016x} final_state_fp={:016x} bounds_fp={:016x} \
             cache=all_invalidated",
            backend.telemetry_name(),
            tag.survivor_row,
            scalar.historical_lower,
            scalar.merged_lower,
            scalar.merged_lower - scalar.historical_lower,
            tag.active_set.rows.len(),
            tag.active_set.state_identity.fingerprint(),
            tag.final_state_identity.fingerprint,
            tag.published_bounds_fingerprint,
        ),
    })
}

fn emit_active_set_scalar_cascade_telemetry(event: ActiveSetScalarCascadeTelemetry<'_>) {
    if !crate::phase_telemetry::phase_telemetry_enabled() {
        return;
    }
    if let Some(line) = active_set_scalar_cascade_telemetry_line_if(true, event) {
        eprintln!("{line}");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_critical_gpu_alpha_on_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    engine: &dyn GemmEngine,
    plan: &CriticalGpuSpecPlan,
    initial_state: &GraphDomainAlphaState,
    adaptive_config: &crate::beta_crown::config::AdaptiveOptConfig,
    bracket_base_lr: Option<f32>,
) -> std::result::Result<CriticalGpuAlphaStepOutput, CriticalGpuAlphaRefusal> {
    if !root_critical_gpu_sound_route_available(Some(engine), plan.deadline) {
        return Err(CriticalGpuAlphaRefusal::Step(
            CriticalGpuAlphaStepRefusal::NoSoundGpuRoute,
        ));
    }
    dispatch_critical_gpu_alpha_lr_bracket(
        bracket_base_lr,
        || {
            run_one_critical_gpu_alpha_step(
                graph,
                input,
                node_bounds,
                engine,
                &plan.spec_matrix,
                plan.deadline,
                initial_state,
                adaptive_config,
            )
        },
        |base_lr| {
            run_critical_gpu_alpha_lr_bracket(
                graph,
                input,
                node_bounds,
                engine,
                &plan.spec_matrix,
                plan.deadline,
                initial_state,
                adaptive_config,
                base_lr,
            )
        },
    )
    .map_err(CriticalGpuAlphaRefusal::Step)
}

type CriticalGpuAlphaAttempt =
    std::result::Result<CriticalGpuAlphaStepOutput, CriticalGpuAlphaRefusal>;

fn flatten_critical_gpu_alpha_factory_result(
    nested: Result<Option<CriticalGpuAlphaAttempt>>,
) -> CriticalGpuAlphaAttempt {
    match nested {
        Ok(Some(inner)) => inner,
        Ok(None) => Err(CriticalGpuAlphaRefusal::FactoryUnavailable),
        Err(error) => {
            debug!(%error, "Critical-row alpha sound f64 factory route errored");
            Err(if error.is_deadline_exceeded() {
                CriticalGpuAlphaRefusal::Step(CriticalGpuAlphaStepRefusal::DeadlineExpired)
            } else {
                CriticalGpuAlphaRefusal::FactoryError
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_critical_gpu_alpha_candidate(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    engine: Option<&dyn GemmEngine>,
    plan: &CriticalGpuSpecPlan,
    initial_state: &GraphDomainAlphaState,
    adaptive_config: &crate::beta_crown::config::AdaptiveOptConfig,
    bracket_base_lr: Option<f32>,
) -> CriticalGpuAlphaRoutedAttempt {
    let backend = select_critical_gpu_spec_backend(engine);
    emit_critical_gpu_alpha_telemetry(CriticalGpuAlphaTelemetry::BackendSelected {
        backend,
        row: plan.row_index,
    });
    let result = match backend {
        CriticalGpuSpecBackend::Local => match engine {
            Some(local_engine) => run_critical_gpu_alpha_on_engine(
                graph,
                input,
                node_bounds,
                local_engine,
                plan,
                initial_state,
                adaptive_config,
                bracket_base_lr,
            ),
            None => Err(CriticalGpuAlphaRefusal::Step(
                CriticalGpuAlphaStepRefusal::NoSoundGpuRoute,
            )),
        },
        CriticalGpuSpecBackend::Factory => flatten_critical_gpu_alpha_factory_result(
            crate::sound_f64_gemm::with_engine_deadline(plan.deadline, |factory_engine| {
                run_critical_gpu_alpha_on_engine(
                    graph,
                    input,
                    node_bounds,
                    factory_engine,
                    plan,
                    initial_state,
                    adaptive_config,
                    bracket_base_lr,
                )
            }),
        ),
    };
    CriticalGpuAlphaRoutedAttempt { backend, result }
}

fn build_critical_gpu_alpha_publication(
    objective_bounds: &[(f32, f32)],
    plan: &CriticalGpuSpecPlan,
    evaluation: CriticalGpuAlphaStepOutput,
) -> std::result::Result<CriticalGpuAlphaPublication, CriticalGpuAlphaRefusal> {
    build_critical_gpu_alpha_publication_with_clock(
        objective_bounds,
        plan,
        evaluation,
        std::time::Instant::now,
    )
}

#[derive(Debug, Clone, Copy)]
struct ValidatedCriticalGpuAlphaPair {
    accepted: CriticalGpuSpecAccepted,
    state_identity: CriticalGpuAlphaStateIdentity,
}

fn validate_critical_gpu_alpha_pair(
    pair: &CriticalGpuAlphaCertifiedPair,
    plan: &CriticalGpuSpecPlan,
    historical: (f32, f32),
    completed_at: std::time::Instant,
    invalid_candidate: CriticalGpuAlphaRefusal,
) -> std::result::Result<ValidatedCriticalGpuAlphaPair, CriticalGpuAlphaRefusal> {
    let recomputed_identity =
        alpha_state_identity(&pair.state).ok_or(CriticalGpuAlphaRefusal::StateIdentityMismatch)?;
    if recomputed_identity != pair.state_identity {
        return Err(CriticalGpuAlphaRefusal::StateIdentityMismatch);
    }
    let accepted = accept_critical_gpu_spec_candidate(plan, historical, &pair.bounds, completed_at)
        .map_err(|reason| match reason {
            CriticalGpuSpecRefusal::CompletedAfterDeadline => {
                CriticalGpuAlphaRefusal::CompletedAfterDeadline
            }
            CriticalGpuSpecRefusal::CandidateDisjoint => CriticalGpuAlphaRefusal::CandidateDisjoint,
            _ => invalid_candidate,
        })?;
    Ok(ValidatedCriticalGpuAlphaPair {
        accepted,
        state_identity: recomputed_identity,
    })
}

fn build_critical_gpu_alpha_publication_with_clock<F>(
    objective_bounds: &[(f32, f32)],
    plan: &CriticalGpuSpecPlan,
    evaluation: CriticalGpuAlphaStepOutput,
    mut now: F,
) -> std::result::Result<CriticalGpuAlphaPublication, CriticalGpuAlphaRefusal>
where
    F: FnMut() -> std::time::Instant,
{
    let bracket = match evaluation.search_provenance.as_ref() {
        Some(provenance)
            if provenance
                .matches_best_candidate(&evaluation.initial, &evaluation.final_candidate) =>
        {
            Some(CriticalGpuAlphaBracketTag {
                selected_ordinal: provenance.selected_ordinal,
                selected_lr: provenance.selected_lr,
                evaluated_candidates: provenance.candidates.len(),
                gradient_replays: provenance.gradient_replays,
            })
        }
        Some(_) => return Err(CriticalGpuAlphaRefusal::SearchProvenanceMismatch),
        None => None,
    };
    let historical = objective_bounds
        .get(plan.row_index)
        .copied()
        .ok_or(CriticalGpuAlphaRefusal::InvalidInitialCandidate)?;
    let initial = validate_critical_gpu_alpha_pair(
        &evaluation.initial,
        plan,
        historical,
        now(),
        CriticalGpuAlphaRefusal::InvalidInitialCandidate,
    )?;
    let final_candidate = validate_critical_gpu_alpha_pair(
        &evaluation.final_candidate,
        plan,
        historical,
        now(),
        CriticalGpuAlphaRefusal::InvalidFinalCandidate,
    )?;

    // This is a paired best-of decision, not a scalar transplant. A regressive
    // Adam step retains the exact initial direct-C enclosure and the exact
    // round-tripped state used to evaluate it. The final state is eligible only
    // under a strict certified improvement over that initial evaluation.
    let select_final = final_candidate.accepted.candidate_lower > initial.accepted.candidate_lower;
    let (selected_pair, selected, selected_kind) = if select_final {
        (
            evaluation.final_candidate,
            final_candidate,
            CriticalGpuAlphaSelectedPair::Final,
        )
    } else {
        (
            evaluation.initial,
            initial,
            CriticalGpuAlphaSelectedPair::Initial,
        )
    };
    let accepted = selected.accepted;
    if accepted.merged_lower <= accepted.historical_lower {
        return Err(CriticalGpuAlphaRefusal::NoCertifiedImprovement);
    }

    // Construct the complete replacement off to the side.  The final clock
    // sample occurs after all validation/allocation and before the pair becomes
    // visible to the caller, so late work can never leak either half.
    let mut published_bounds = objective_bounds.to_vec();
    published_bounds[plan.row_index].0 = accepted.merged_lower;
    if now() >= plan.deadline {
        return Err(CriticalGpuAlphaRefusal::CompletedAfterDeadline);
    }
    let transport_tag = CriticalGpuAlphaTransportTag {
        row_index: plan.row_index,
        historical_lower: accepted.historical_lower,
        initial_lower: initial.accepted.candidate_lower,
        final_lower: final_candidate.accepted.candidate_lower,
        merged_lower: accepted.merged_lower,
        selected_pair: selected_kind,
        state_identity: selected.state_identity,
        bracket,
    };
    Ok(CriticalGpuAlphaPublication {
        pair: CriticalGpuAlphaRootPair {
            objective_bounds: published_bounds,
            alpha_state: selected_pair.state,
            transport_tag: Some(transport_tag),
            active_set_transport_tag: None,
            active_set_scalar_cascade_transport_tag: None,
        },
        accepted,
    })
}

/// The post-tightening adaptive coefficient backward is an optional quality
/// candidate, never a prerequisite for a sound root bound. Keep it opt-in:
/// deep image DAGs can spend the entire remaining BaB slice materializing its
/// dense fallback even after the historical GPU root candidate has completed.
///
/// `NY_ROOT_SKIP_ADAPTIVE_SPEC=1` remains a compatibility override for old
/// probes. Requiring the exact value `1` for the positive gate keeps malformed
/// deployment values fail-closed on the bounded historical path.
fn root_adaptive_spec_enabled_from_values(enable: Option<&str>, skip: Option<&str>) -> bool {
    matches!(enable, Some("1")) && !matches!(skip, Some("1"))
}

fn root_adaptive_spec_enabled() -> bool {
    root_adaptive_spec_enabled_from_values(
        std::env::var("NY_ROOT_ADAPTIVE_SPEC").ok().as_deref(),
        std::env::var("NY_ROOT_SKIP_ADAPTIVE_SPEC").ok().as_deref(),
    )
}

#[cfg(test)]
mod post_tightening_alpha_tests {
    use super::{
        best_of_sound_root_spec_bounds, root_adaptive_spec_enabled_from_values,
        run_adaptive_root_spec_candidate,
    };
    use crate::bounds::GraphAlphaState;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::{GraphNetwork, GraphNode};
    use ndarray::{arr1, arr2, Array1};
    use ny_tensor::BoundedTensor;

    fn stale_alpha_fixture() -> (GraphNetwork, BoundedTensor, Vec<f32>, Vec<f32>) {
        // All four hidden intervals cross zero. The adaptive slopes are
        // [1, 1, 0, 1]; a stale all-zero alpha loses >1.6 lower-margin units.
        let hidden_w = vec![2.361_429_2, 1.780_559_5, 1.406_410_1, 2.439_561_8];
        let hidden_b = vec![1.051_541_9, 1.158_990_5, -0.584_852_1, 1.923_906_3];
        let output_w = vec![2.771_405_7, -2.032_892, 1.524_024_4, 1.290_905_4];

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "hidden",
            Layer::Linear(
                LinearLayer::new(
                    ndarray::Array2::from_shape_vec((4, 1), hidden_w.clone()).unwrap(),
                    Some(Array1::from_vec(hidden_b.clone())),
                )
                .unwrap(),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["hidden".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(
                LinearLayer::new(
                    ndarray::Array2::from_shape_vec((1, 4), output_w).unwrap(),
                    None,
                )
                .unwrap(),
            ),
            vec!["relu".to_string()],
        ));
        graph.set_output("output");
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        (graph, input, hidden_w, hidden_b)
    }

    #[test]
    fn best_of_root_candidates_intersects_each_objective_row() {
        let adaptive = BoundedTensor::new(
            arr1(&[0.0_f32, 1.0, 2.0]).into_dyn(),
            arr1(&[10.0_f32, 11.0, 12.0]).into_dyn(),
        )
        .unwrap();
        let historical = BoundedTensor::new(
            arr1(&[1.0_f32, 0.0, 4.0]).into_dyn(),
            arr1(&[9.0_f32, 8.0, 13.0]).into_dyn(),
        )
        .unwrap();
        let best = best_of_sound_root_spec_bounds(adaptive, &historical);
        assert_eq!(best.lower(), &arr1(&[1.0_f32, 1.0, 4.0]).into_dyn());
        assert_eq!(best.upper(), &arr1(&[9.0_f32, 8.0, 12.0]).into_dyn());

        let adaptive = BoundedTensor::new(
            arr1(&[0.0_f32, 1.0, 2.0]).into_dyn(),
            arr1(&[10.0_f32, 11.0, 12.0]).into_dyn(),
        )
        .unwrap();
        let partially_disjoint = BoundedTensor::new(
            arr1(&[1.0_f32, 20.0, 4.0]).into_dyn(),
            arr1(&[9.0_f32, 21.0, 13.0]).into_dyn(),
        )
        .unwrap();
        let best = best_of_sound_root_spec_bounds(adaptive, &partially_disjoint);
        assert_eq!(best.lower(), &arr1(&[1.0_f32, 1.0, 4.0]).into_dyn());
        assert_eq!(best.upper(), &arr1(&[9.0_f32, 11.0, 12.0]).into_dyn());
    }

    #[test]
    fn adaptive_root_spec_is_exact_opt_in_with_skip_override() {
        assert!(!root_adaptive_spec_enabled_from_values(None, None));
        assert!(!root_adaptive_spec_enabled_from_values(Some("0"), None));
        assert!(!root_adaptive_spec_enabled_from_values(Some("true"), None));
        assert!(root_adaptive_spec_enabled_from_values(Some("1"), None));
        assert!(root_adaptive_spec_enabled_from_values(Some("1"), Some("0")));
        assert!(!root_adaptive_spec_enabled_from_values(
            Some("1"),
            Some("1")
        ));
    }

    #[test]
    fn post_tightening_best_of_stale_alpha_cannot_worsen_adaptive() {
        let (graph, input, hidden_w, hidden_b) = stale_alpha_fixture();
        let node_bounds = graph.collect_node_bounds(&input).unwrap();
        let spec = arr2(&[[1.0_f32]]);

        let (adaptive, _) = run_adaptive_root_spec_candidate(
            &graph,
            &input,
            &spec,
            None,
            &node_bounds,
            None,
            None,
            None,
        )
        .unwrap();
        let mut stale = GraphAlphaState::new();
        stale.alphas.insert("relu".to_string(), Array1::zeros(4));
        stale
            .alphas_upper
            .insert("relu".to_string(), Array1::zeros(4));
        stale
            .unstable_mask
            .insert("relu".to_string(), Array1::from_elem(4, true));
        let (stale_bounds, _) = run_adaptive_root_spec_candidate(
            &graph,
            &input,
            &spec,
            None,
            &node_bounds,
            Some(&stale),
            None,
            None,
        )
        .unwrap();

        let adaptive_lower = adaptive.lower()[[0]];
        let stale_lower = stale_bounds.lower()[[0]];
        assert!(
            adaptive_lower > stale_lower + 1.5,
            "fixture must discriminate adaptive from stale alpha: adaptive={adaptive_lower}, stale={stale_lower}"
        );
        let best = best_of_sound_root_spec_bounds(adaptive.clone(), &stale_bounds);
        assert!(best.lower()[[0]] >= adaptive_lower);
        assert!(best.upper()[[0]] <= adaptive.upper()[[0]]);

        let disjoint = BoundedTensor::new(
            arr1(&[adaptive.upper()[[0]] + 1.0]).into_dyn(),
            arr1(&[adaptive.upper()[[0]] + 2.0]).into_dyn(),
        )
        .unwrap();
        let preserved = best_of_sound_root_spec_bounds(adaptive.clone(), &disjoint);
        assert_eq!(preserved.lower(), adaptive.lower());
        assert_eq!(preserved.upper(), adaptive.upper());

        // The sound intersection must still enclose the concrete network.
        let output_w = [2.771_405_7, -2.032_892, 1.524_024_4, 1.290_905_4];
        for step in 0..=100 {
            let x = -1.0 + 2.0 * step as f32 / 100.0;
            let y: f32 = hidden_w
                .iter()
                .zip(&hidden_b)
                .zip(output_w)
                .map(|((&weight, &bias), output)| output * (weight * x + bias).max(0.0))
                .sum();
            assert!(
                y >= best.lower()[[0]] - 1e-5 && y <= best.upper()[[0]] + 1e-5,
                "best-of enclosure missed y={y} at x={x}: [{}, {}]",
                best.lower()[[0]],
                best.upper()[[0]]
            );
        }
    }
}

#[cfg(test)]
mod critical_gpu_spec_tests {
    use super::{
        accept_critical_gpu_spec_candidate, apply_critical_gpu_spec_lower_only,
        apply_critical_gpu_spec_lower_only_with_clock, build_critical_gpu_spec_plan,
        critical_gpu_spec_telemetry_line_if, flatten_critical_gpu_spec_factory_result,
        root_critical_gpu_sound_route_available, root_critical_gpu_spec_enabled_from_value,
        select_critical_gpu_spec_backend, CriticalGpuSpecAccepted, CriticalGpuSpecBackend,
        CriticalGpuSpecPlan, CriticalGpuSpecRefusal, CriticalGpuSpecTelemetry,
        ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE, ROOT_CRITICAL_GPU_SPEC_MAX_RUNTIME,
    };
    use ndarray::{arr1, arr2, Array2};
    use ny_core::{GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NyError, Result};
    use ny_tensor::BoundedTensor;
    use std::time::{Duration, Instant};

    struct FakeGpuEngine {
        sound: bool,
        dedicated_ats_one_row: bool,
    }

    impl GemmEngine for FakeGpuEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("fake engine".into()))
        }

        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }
    }

    impl GpuCrownBackward for FakeGpuEngine {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp("fake engine".into()))
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.sound
        }

        fn provides_deadline_bounded_single_row_resnet_sound(&self) -> bool {
            self.dedicated_ats_one_row
        }
    }

    fn candidate(rows: &[(f32, f32)]) -> BoundedTensor {
        BoundedTensor::new(
            arr1(&rows.iter().map(|&(lower, _)| lower).collect::<Vec<_>>()).into_dyn(),
            arr1(&rows.iter().map(|&(_, upper)| upper).collect::<Vec<_>>()).into_dyn(),
        )
        .unwrap()
    }

    fn plan(row_index: usize, deadline: Instant) -> CriticalGpuSpecPlan {
        CriticalGpuSpecPlan {
            row_index,
            spec_matrix: Array2::zeros((1, 2)),
            deadline,
        }
    }

    #[test]
    fn critical_gpu_spec_gate_is_exact_opt_in() {
        assert!(!root_critical_gpu_spec_enabled_from_value(None));
        assert!(!root_critical_gpu_spec_enabled_from_value(Some("0")));
        assert!(!root_critical_gpu_spec_enabled_from_value(Some("true")));
        assert!(!root_critical_gpu_spec_enabled_from_value(Some(" 1")));
        assert!(root_critical_gpu_spec_enabled_from_value(Some("1")));
        assert!(!root_critical_gpu_sound_route_available(
            None,
            Instant::now() + Duration::from_secs(1)
        ));
    }

    #[test]
    fn critical_gpu_spec_routes_wgpu_like_passed_engine_to_ats_factory() {
        let wgpu_like = FakeGpuEngine {
            sound: true,
            dedicated_ats_one_row: false,
        };
        let ats_factory_engine = FakeGpuEngine {
            sound: true,
            dedicated_ats_one_row: true,
        };
        let unsound_claim = FakeGpuEngine {
            sound: false,
            dedicated_ats_one_row: true,
        };

        assert_eq!(
            select_critical_gpu_spec_backend(Some(&wgpu_like)),
            CriticalGpuSpecBackend::Factory,
            "the CIFAR WGPU engine lacks the dedicated ATS one-row contract"
        );
        assert_eq!(
            select_critical_gpu_spec_backend(Some(&ats_factory_engine)),
            CriticalGpuSpecBackend::Local,
            "an exact passed ATS engine stays call-local"
        );
        assert_eq!(
            select_critical_gpu_spec_backend(Some(&unsound_claim)),
            CriticalGpuSpecBackend::Factory,
            "the dedicated flag alone cannot bypass the sound-GPU contract"
        );
        assert_eq!(
            select_critical_gpu_spec_backend(None),
            CriticalGpuSpecBackend::Factory
        );
    }

    #[test]
    fn critical_gpu_spec_factory_nested_result_is_fail_closed() {
        let accepted =
            flatten_critical_gpu_spec_factory_result(Ok(Some(Ok(CriticalGpuSpecAccepted {
                historical_lower: -1.0,
                candidate_lower: -0.25,
                merged_lower: -0.25,
            }))));
        let accepted = match accepted {
            Ok(accepted) => accepted,
            Err(reason) => panic!("fully successful nested result refused: {reason:?}"),
        };
        assert_eq!(accepted.merged_lower, -0.25);

        assert_eq!(
            flatten_critical_gpu_spec_factory_result(Ok(Some(Err(
                CriticalGpuSpecRefusal::NoSoundGpuRoute
            ))))
            .err(),
            Some(CriticalGpuSpecRefusal::NoSoundGpuRoute),
            "an exact factory-engine capability refusal must stay a refusal"
        );
        assert_eq!(
            flatten_critical_gpu_spec_factory_result(Ok(None)).err(),
            Some(CriticalGpuSpecRefusal::GpuRouteUnavailable),
            "an absent factory engine must not publish"
        );
        assert_eq!(
            flatten_critical_gpu_spec_factory_result(Err(NyError::DeadlineExceeded(
                "test deadline".into()
            )))
            .err(),
            Some(CriticalGpuSpecRefusal::CompletedAfterDeadline),
            "factory admission deadlines must retain deadline provenance"
        );
        assert_eq!(
            flatten_critical_gpu_spec_factory_result(Err(NyError::InvalidSpec(
                "test failure".into()
            )))
            .err(),
            Some(CriticalGpuSpecRefusal::GpuRouteError),
            "ordinary factory admission errors remain route errors"
        );
    }

    #[test]
    fn critical_gpu_spec_telemetry_is_exact_gated_and_stable() {
        let accepted = CriticalGpuSpecAccepted {
            historical_lower: -1.0,
            candidate_lower: -0.25,
            merged_lower: -0.25,
        };
        let selected = CriticalGpuSpecTelemetry::BackendSelected {
            backend: CriticalGpuSpecBackend::Factory,
            row: 7,
        };
        assert_eq!(
            critical_gpu_spec_telemetry_line_if(false, selected),
            None,
            "default-unset telemetry path must produce no marker"
        );
        assert_eq!(
            critical_gpu_spec_telemetry_line_if(true, selected).as_deref(),
            Some("[critical-gpu-spec] status=selected backend=factory row=7")
        );
        assert_eq!(
            critical_gpu_spec_telemetry_line_if(
                true,
                CriticalGpuSpecTelemetry::CandidateRefused {
                    backend: CriticalGpuSpecBackend::Factory,
                    row: 7,
                    reason: CriticalGpuSpecRefusal::NoSoundGpuRoute,
                },
            )
            .as_deref(),
            Some(
                "[critical-gpu-spec] status=refused backend=factory row=7 \
                 reason=no_sound_gpu_route"
            )
        );
        assert_eq!(
            critical_gpu_spec_telemetry_line_if(
                true,
                CriticalGpuSpecTelemetry::PlanRefused {
                    reason: CriticalGpuSpecRefusal::InsufficientHeadroom,
                },
            )
            .as_deref(),
            Some(
                "[critical-gpu-spec] status=refused backend=unselected row=none \
                 reason=insufficient_headroom"
            )
        );
        assert_eq!(
            critical_gpu_spec_telemetry_line_if(
                true,
                CriticalGpuSpecTelemetry::Accepted {
                    backend: CriticalGpuSpecBackend::Factory,
                    row: 7,
                    accepted: &accepted,
                },
            )
            .as_deref(),
            Some(
                "[critical-gpu-spec] status=accepted backend=factory row=7 \
                 historical_lower=-1 candidate_lower=-0.25 merged_lower=-0.25 lift=0.75"
            )
        );
    }

    #[test]
    fn critical_gpu_spec_policy_selects_exactly_one_unresolved_row() {
        let now = Instant::now();
        let historical = vec![(1.0, 2.0), (-1.0, 3.0), (5.0, 6.0)];
        let spec = arr2(&[[1.0_f32, 0.0], [0.25_f32, -0.75], [-1.0_f32, 0.0]]);
        let thresholds = [0.0_f32, 0.0, 4.0];
        let plan = build_critical_gpu_spec_plan(
            true,
            false,
            None,
            &historical,
            &spec,
            &thresholds,
            now,
            Some(now + ROOT_CRITICAL_GPU_SPEC_MAX_RUNTIME + ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE),
        )
        .expect("one unresolved row with full headroom must be admitted");

        assert_eq!(plan.row_index, 1);
        assert_eq!(plan.spec_matrix.shape(), &[1, 2]);
        assert_eq!(plan.spec_matrix.row(0), spec.row(1));
        assert_eq!(plan.deadline, now + ROOT_CRITICAL_GPU_SPEC_MAX_RUNTIME);
    }

    #[test]
    fn critical_gpu_spec_admits_selected_dense_head_with_zero_tightening() {
        let now = Instant::now();
        let dense_head_stage_selected = true;
        let tightened_elements = 0_usize;
        assert_eq!(tightened_elements, 0);

        let admitted = build_critical_gpu_spec_plan(
            dense_head_stage_selected,
            false,
            None,
            &[(-1.0, 3.0)],
            &arr2(&[[1.0_f32, -1.0]]),
            &[0.0],
            now,
            Some(now + ROOT_CRITICAL_GPU_SPEC_MAX_RUNTIME + ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE),
        );
        assert!(
            admitted.is_ok(),
            "certified bootstrap bounds remain eligible when the selected dense-head pass changes zero elements"
        );
    }

    #[test]
    fn critical_gpu_spec_policy_refuses_every_noncritical_surface() {
        let now = Instant::now();
        let one_active = vec![(2.0, 3.0), (-1.0, 4.0)];
        let no_active = vec![(2.0, 3.0), (1.0, 4.0)];
        let two_active = vec![(-2.0, 3.0), (-1.0, 4.0)];
        let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0_f32, 0.0];
        let global =
            Some(now + ROOT_CRITICAL_GPU_SPEC_MAX_RUNTIME + ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE);
        let build = |dense_head_selected, conjunctive, truncation, bounds, deadline| {
            build_critical_gpu_spec_plan(
                dense_head_selected,
                conjunctive,
                truncation,
                bounds,
                &spec,
                &thresholds,
                now,
                deadline,
            )
            .map(|_| ())
        };

        assert_eq!(
            build(false, false, None, &one_active, global),
            Err(CriticalGpuSpecRefusal::DenseHeadStageNotSelected)
        );
        assert_eq!(
            build(true, true, None, &one_active, global),
            Err(CriticalGpuSpecRefusal::ConjunctiveProperty)
        );
        assert_eq!(
            build(true, false, Some(3), &one_active, global),
            Err(CriticalGpuSpecRefusal::TruncatedBackward)
        );
        assert_eq!(
            build(true, false, None, &one_active[..1], global),
            Err(CriticalGpuSpecRefusal::InvalidInput)
        );
        assert_eq!(
            build(true, false, None, &no_active, global),
            Err(CriticalGpuSpecRefusal::NoUnresolvedRow)
        );
        assert_eq!(
            build(true, false, None, &two_active, global),
            Err(CriticalGpuSpecRefusal::TooManyUnresolvedRows)
        );
        assert_eq!(
            build(
                true,
                false,
                None,
                &one_active,
                Some(
                    (now + ROOT_CRITICAL_GPU_SPEC_MAX_RUNTIME + ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE)
                        .checked_sub(Duration::from_nanos(1))
                        .expect("one nanosecond must be representable"),
                ),
            ),
            Err(CriticalGpuSpecRefusal::InsufficientHeadroom)
        );
    }

    #[test]
    fn critical_gpu_spec_accepts_only_scalar_lower() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let one = plan(1, deadline);
        let accepted = accept_critical_gpu_spec_candidate(
            &one,
            (-1.0, 3.0),
            &candidate(&[(-0.25, 999.0)]),
            Instant::now(),
        )
        .expect("finite overlapping scalar lower bound must be accepted");

        assert_eq!(accepted.historical_lower, -1.0);
        assert_eq!(accepted.candidate_lower, -0.25);
        assert_eq!(accepted.merged_lower, -0.25);

        let mut objective_bounds = vec![(1.0_f32, 2.0_f32), (-1.0, 3.0), (5.0, 6.0)];
        let before = objective_bounds
            .iter()
            .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
            .collect::<Vec<_>>();
        assert!(
            apply_critical_gpu_spec_lower_only(&mut objective_bounds, 1, &accepted, &one,).is_ok()
        );
        assert_eq!(objective_bounds[1].0, -0.25);
        for (row, &(lower_bits, upper_bits)) in before.iter().enumerate() {
            assert_eq!(
                objective_bounds[row].1.to_bits(),
                upper_bits,
                "no upper endpoint may be published from the candidate"
            );
            if row != 1 {
                assert_eq!(
                    objective_bounds[row].0.to_bits(),
                    lower_bits,
                    "every non-selected row must remain byte-identical"
                );
            }
        }
    }

    #[test]
    fn critical_gpu_spec_faults_never_produce_a_lower_bound() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let one = plan(1, deadline);

        assert_eq!(
            accept_critical_gpu_spec_candidate(
                &one,
                (-1.0, 3.0),
                &candidate(&[(-0.5, 1.0), (-0.25, 2.0)]),
                Instant::now(),
            )
            .err(),
            Some(CriticalGpuSpecRefusal::InvalidCandidate)
        );
        let non_finite = BoundedTensor::new_allow_infinite(
            arr1(&[f32::NEG_INFINITY]).into_dyn(),
            arr1(&[1.0_f32]).into_dyn(),
        )
        .unwrap();
        assert_eq!(
            accept_critical_gpu_spec_candidate(&one, (-1.0, 3.0), &non_finite, Instant::now(),)
                .err(),
            Some(CriticalGpuSpecRefusal::InvalidCandidate)
        );
        assert_eq!(
            accept_critical_gpu_spec_candidate(
                &one,
                (-1.0, 3.0),
                &candidate(&[(4.0, 5.0)]),
                Instant::now(),
            )
            .err(),
            Some(CriticalGpuSpecRefusal::CandidateDisjoint)
        );
        assert_eq!(
            accept_critical_gpu_spec_candidate(
                &one,
                (-1.0, 3.0),
                &candidate(&[(-4.0, -2.0)]),
                Instant::now(),
            )
            .err(),
            Some(CriticalGpuSpecRefusal::CandidateDisjoint)
        );
        assert_eq!(
            accept_critical_gpu_spec_candidate(
                &one,
                (-1.0, 3.0),
                &candidate(&[(-0.25, 2.0)]),
                deadline + Duration::from_nanos(1),
            )
            .err(),
            Some(CriticalGpuSpecRefusal::CompletedAfterDeadline)
        );
    }

    #[test]
    fn critical_gpu_spec_deadline_boundary_is_fail_closed() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let one = plan(1, deadline);
        let bounded = candidate(&[(-0.25, 2.0)]);

        assert!(
            accept_critical_gpu_spec_candidate(
                &one,
                (-1.0, 3.0),
                &bounded,
                deadline
                    .checked_sub(Duration::from_nanos(1))
                    .expect("one nanosecond must be representable"),
            )
            .is_ok(),
            "a candidate completed strictly before its private deadline remains eligible"
        );
        for completed_at in [deadline, deadline + Duration::from_nanos(1)] {
            assert_eq!(
                accept_critical_gpu_spec_candidate(&one, (-1.0, 3.0), &bounded, completed_at,)
                    .err(),
                Some(CriticalGpuSpecRefusal::CompletedAfterDeadline),
                "completion at or after the private deadline must refuse publication"
            );
        }
    }

    #[test]
    fn critical_gpu_spec_publication_rechecks_deadline_without_mutation() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let one = plan(1, deadline);
        let accepted = accept_critical_gpu_spec_candidate(
            &one,
            (-1.0, 3.0),
            &candidate(&[(-0.25, 2.0)]),
            deadline
                .checked_sub(Duration::from_nanos(2))
                .expect("two nanoseconds must be representable"),
        )
        .expect("computation completed before the private deadline");

        let mut before_deadline = vec![(1.0_f32, 2.0_f32), (-1.0, 3.0)];
        assert!(
            apply_critical_gpu_spec_lower_only_with_clock(
                &mut before_deadline,
                1,
                &accepted,
                &one,
                || {
                    deadline
                        .checked_sub(Duration::from_nanos(1))
                        .expect("one nanosecond must be representable")
                },
            )
            .is_ok(),
            "publication strictly before the private deadline remains eligible"
        );
        assert_eq!(before_deadline[1], (-0.25, 3.0));

        for attempted_at in [deadline, deadline + Duration::from_nanos(1)] {
            let mut expired = vec![(1.0_f32, 2.0_f32), (-1.0, 3.0)];
            let before = expired
                .iter()
                .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
                .collect::<Vec<_>>();
            assert_eq!(
                apply_critical_gpu_spec_lower_only_with_clock(
                    &mut expired,
                    1,
                    &accepted,
                    &one,
                    || attempted_at,
                )
                .err(),
                Some(CriticalGpuSpecRefusal::CompletedAfterDeadline)
            );
            assert_eq!(
                expired
                    .iter()
                    .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
                    .collect::<Vec<_>>(),
                before,
                "expired publication must leave every endpoint byte-identical"
            );
        }
    }
}

#[cfg(test)]
mod critical_gpu_alpha_tests {
    use super::{
        build_critical_gpu_alpha_publication_with_clock, build_critical_gpu_spec_plan_with_runtime,
        critical_gpu_alpha_preserves_per_disjunct, critical_gpu_alpha_telemetry_line_if,
        dispatch_critical_gpu_alpha_lr_bracket, flatten_critical_gpu_alpha_factory_result,
        root_critical_gpu_alpha_enabled_from_value, root_critical_gpu_spec_enabled_from_value,
        with_critical_gpu_alpha_gate, CriticalGpuAlphaRefusal, CriticalGpuAlphaSelectedPair,
        CriticalGpuAlphaTelemetry, CriticalGpuSpecBackend, CriticalGpuSpecPlan,
        ROOT_CRITICAL_GPU_ALPHA_MAX_RUNTIME, ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE,
    };
    use crate::batched_domain::CachedLinearBounds;
    use crate::beta_crown::branching::GraphNeuronConstraint;
    use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
    use crate::beta_crown::engine::graph::multi_objective::critical_gpu_alpha::{
        alpha_state_identity, critical_gpu_alpha_lr_bracket_enabled_from_value,
        CriticalGpuAlphaCandidateTrace, CriticalGpuAlphaCertifiedPair,
        CriticalGpuAlphaSearchProvenance, CriticalGpuAlphaStepOutput, CriticalGpuAlphaStepRefusal,
    };
    use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};
    use crate::layers::{Layer, ReLULayer};
    use crate::network::{GraphNetwork, GraphNode};
    use ndarray::{arr1, arr2, Array2};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    fn candidate(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new_allow_infinite(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
            .expect("scalar candidate shape")
    }

    #[test]
    fn critical_gpu_alpha_factory_preserves_deadline_refusal() {
        assert_eq!(
            flatten_critical_gpu_alpha_factory_result(Err(NyError::DeadlineExceeded(
                "test deadline".into()
            )))
            .err(),
            Some(CriticalGpuAlphaRefusal::Step(
                CriticalGpuAlphaStepRefusal::DeadlineExpired
            ))
        );
        assert_eq!(
            flatten_critical_gpu_alpha_factory_result(Err(NyError::InvalidSpec(
                "test failure".into()
            )))
            .err(),
            Some(CriticalGpuAlphaRefusal::FactoryError)
        );
    }

    fn plan(row_index: usize, deadline: Instant) -> CriticalGpuSpecPlan {
        CriticalGpuSpecPlan {
            row_index,
            spec_matrix: Array2::zeros((1, 2)),
            deadline,
        }
    }

    fn just_before(instant: Instant) -> Instant {
        instant
            .checked_sub(Duration::from_nanos(1))
            .expect("one nanosecond before a future test deadline must be representable")
    }

    fn alpha_state(first: f32, second: f32) -> GraphDomainAlphaState {
        let mut state = GraphDomainAlphaState::empty();
        state.insert("relu0".into(), 0, AlphaNeuronState::new(first));
        state.insert("relu0".into(), 1, AlphaNeuronState::new(second));
        state
    }

    fn certified_pair(
        lower: f32,
        upper: f32,
        first_alpha: f32,
        second_alpha: f32,
    ) -> CriticalGpuAlphaCertifiedPair {
        let state = alpha_state(first_alpha, second_alpha);
        let state_identity = alpha_state_identity(&state).expect("valid alpha state");
        CriticalGpuAlphaCertifiedPair {
            bounds: candidate(lower, upper),
            state,
            state_identity,
        }
    }

    fn evaluation(lower: f32, upper: f32) -> CriticalGpuAlphaStepOutput {
        evaluation_with(-1.25, 1.75, lower, upper)
    }

    fn evaluation_with(
        initial_lower: f32,
        initial_upper: f32,
        final_lower: f32,
        final_upper: f32,
    ) -> CriticalGpuAlphaStepOutput {
        CriticalGpuAlphaStepOutput {
            initial: certified_pair(initial_lower, initial_upper, 0.125, 0.625),
            final_candidate: certified_pair(final_lower, final_upper, 0.25, 0.75),
            search_provenance: None,
        }
    }

    fn bracket_evaluation(initial_lower: f32, final_lower: f32) -> CriticalGpuAlphaStepOutput {
        let mut evaluation = evaluation_with(initial_lower, 1.75, final_lower, 1.5);
        let final_identity = evaluation.final_candidate.state_identity;
        evaluation.search_provenance = Some(CriticalGpuAlphaSearchProvenance {
            base_lr: 0.1,
            candidates: vec![
                CriticalGpuAlphaCandidateTrace {
                    ordinal: 0,
                    adam_t: 1,
                    alpha_lr: 0.1_f32 * 0.3,
                    lower: final_lower - 0.1,
                    lift_from_initial: final_lower - 0.1 - initial_lower,
                    state_identity: final_identity,
                },
                CriticalGpuAlphaCandidateTrace {
                    ordinal: 1,
                    adam_t: 1,
                    alpha_lr: 0.1,
                    lower: final_lower,
                    lift_from_initial: final_lower - initial_lower,
                    state_identity: final_identity,
                },
                CriticalGpuAlphaCandidateTrace {
                    ordinal: 2,
                    adam_t: 1,
                    alpha_lr: 0.2,
                    lower: final_lower - 0.2,
                    lift_from_initial: final_lower - 0.2 - initial_lower,
                    state_identity: final_identity,
                },
            ],
            selected_ordinal: 1,
            selected_lr: 0.1,
            gradient_replays: 1,
        });
        evaluation
    }

    #[test]
    fn critical_gpu_alpha_gate_is_exact_and_off_makes_zero_calls() {
        assert!(!root_critical_gpu_alpha_enabled_from_value(None));
        assert!(!root_critical_gpu_alpha_enabled_from_value(Some("0")));
        assert!(!root_critical_gpu_alpha_enabled_from_value(Some("true")));
        assert!(!root_critical_gpu_alpha_enabled_from_value(Some(" 1")));
        assert!(root_critical_gpu_alpha_enabled_from_value(Some("1")));

        static CALLS: AtomicUsize = AtomicUsize::new(0);
        let result = with_critical_gpu_alpha_gate(false, || {
            CALLS.fetch_add(1, Ordering::SeqCst);
            7
        });
        assert_eq!(result, None);
        assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn critical_gpu_alpha_lr_bracket_is_nested_and_dispatches_only_one_body() {
        for malformed in [None, Some("0"), Some("true"), Some(" 1"), Some("1 ")] {
            assert!(!critical_gpu_alpha_lr_bracket_enabled_from_value(malformed));
        }
        assert!(critical_gpu_alpha_lr_bracket_enabled_from_value(Some("1")));

        let nested_calls = Cell::new(0);
        assert_eq!(
            with_critical_gpu_alpha_gate(false, || {
                nested_calls.set(nested_calls.get() + 1);
                critical_gpu_alpha_lr_bracket_enabled_from_value(Some("1"))
            }),
            None
        );
        assert_eq!(
            nested_calls.get(),
            0,
            "parent-off must not inspect or dispatch the nested lane"
        );

        let sealed_calls = Cell::new(0);
        let bracket_calls = Cell::new(0);
        let selected = dispatch_critical_gpu_alpha_lr_bracket(
            None,
            || {
                sealed_calls.set(sealed_calls.get() + 1);
                "sealed"
            },
            |_| {
                bracket_calls.set(bracket_calls.get() + 1);
                "bracket"
            },
        );
        assert_eq!(selected, "sealed");
        assert_eq!(sealed_calls.get(), 1);
        assert_eq!(bracket_calls.get(), 0);

        let selected = dispatch_critical_gpu_alpha_lr_bracket(
            Some(0.1),
            || {
                sealed_calls.set(sealed_calls.get() + 1);
                "sealed"
            },
            |base_lr| {
                assert_eq!(base_lr.to_bits(), 0.1_f32.to_bits());
                bracket_calls.set(bracket_calls.get() + 1);
                "bracket"
            },
        );
        assert_eq!(selected, "bracket");
        assert_eq!(sealed_calls.get(), 1);
        assert_eq!(bracket_calls.get(), 1);
    }

    #[test]
    fn critical_gpu_alpha_gate_presence_does_not_suppress_existing_spec_control() {
        assert!(root_critical_gpu_alpha_enabled_from_value(Some("1")));
        assert!(
            root_critical_gpu_spec_enabled_from_value(Some("1")),
            "the established critical-spec condition depends only on its own exact gate"
        );
    }

    #[test]
    fn critical_gpu_alpha_refusal_preserves_per_disjunct_control_surface() {
        assert!(
            critical_gpu_alpha_preserves_per_disjunct(None),
            "plan, candidate, and publication refusals install no tag and must retain per-disjunct alpha"
        );
    }

    #[test]
    fn critical_gpu_alpha_plan_has_exact_two_second_slice_and_twelve_second_reserve() {
        let now = Instant::now();
        let historical = [(1.0_f32, 2.0_f32), (-1.0, 3.0)];
        let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0_f32, 0.0];
        let admitted = build_critical_gpu_spec_plan_with_runtime(
            true,
            false,
            None,
            &historical,
            &spec,
            &thresholds,
            now,
            Some(now + ROOT_CRITICAL_GPU_ALPHA_MAX_RUNTIME + ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE),
            ROOT_CRITICAL_GPU_ALPHA_MAX_RUNTIME,
        )
        .expect("exact headroom boundary is admissible");
        assert_eq!(admitted.deadline, now + Duration::from_secs(2));
        assert_eq!(admitted.row_index, 1);

        let refused = build_critical_gpu_spec_plan_with_runtime(
            true,
            false,
            None,
            &historical,
            &spec,
            &thresholds,
            now,
            Some(just_before(
                now + ROOT_CRITICAL_GPU_ALPHA_MAX_RUNTIME + ROOT_CRITICAL_GPU_SPEC_BAB_RESERVE,
            )),
            ROOT_CRITICAL_GPU_ALPHA_MAX_RUNTIME,
        );
        assert!(refused.is_err());
    }

    #[test]
    fn critical_gpu_alpha_publication_is_one_atomic_tagged_pair() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let plan = plan(1, deadline);
        let historical = vec![(2.0_f32, 3.0_f32), (-1.0, 2.0), (4.0, 5.0)];
        let before_bits: Vec<_> = historical
            .iter()
            .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
            .collect();
        let publication = build_critical_gpu_alpha_publication_with_clock(
            &historical,
            &plan,
            evaluation(-0.25, 1.5),
            || just_before(deadline),
        )
        .expect("finite overlapping strict lift must publish");
        let tag = publication
            .pair
            .transport_tag
            .expect("publication must carry its state tag");
        assert_eq!(tag.row_index, 1);
        assert_eq!(tag.final_lower, -0.25);
        assert_eq!(tag.merged_lower, -0.25);
        assert_eq!(tag.selected_pair, CriticalGpuAlphaSelectedPair::Final);
        assert!(
            !critical_gpu_alpha_preserves_per_disjunct(Some(&tag)),
            "only an installed authoritative pair suppresses per-disjunct alpha"
        );
        assert_eq!(
            alpha_state_identity(&publication.pair.alpha_state),
            Some(tag.state_identity)
        );
        assert_eq!(publication.pair.objective_bounds[1], (-0.25, 2.0));
        for row in [0, 2] {
            assert_eq!(
                (
                    publication.pair.objective_bounds[row].0.to_bits(),
                    publication.pair.objective_bounds[row].1.to_bits(),
                ),
                before_bits[row],
                "unselected rows must remain bit-exact"
            );
        }
    }

    #[test]
    fn critical_gpu_alpha_best_of_keeps_initial_pair_when_adam_regresses() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let evaluation = evaluation_with(-0.49, 1.5, -0.9, 1.6);
        let initial_identity = evaluation.initial.state_identity;
        let final_identity = evaluation.final_candidate.state_identity;
        assert_ne!(initial_identity, final_identity);

        let publication = build_critical_gpu_alpha_publication_with_clock(
            &[(-1.017_f32, 2.0_f32)],
            &plan(0, deadline),
            evaluation,
            || just_before(deadline),
        )
        .expect("the already-certified initial pair beats history");
        let tag = publication.pair.transport_tag.expect("tagged pair");
        assert_eq!(tag.selected_pair, CriticalGpuAlphaSelectedPair::Initial);
        assert_eq!(tag.initial_lower, -0.49);
        assert_eq!(tag.final_lower, -0.9);
        assert_eq!(tag.merged_lower, -0.49);
        assert_eq!(tag.state_identity, initial_identity);
        assert_eq!(
            alpha_state_identity(&publication.pair.alpha_state),
            Some(initial_identity),
            "the initial direct-C bound must transport only its exact initial state"
        );
        assert_eq!(publication.pair.objective_bounds[0].0, -0.49);
    }

    #[test]
    fn critical_gpu_alpha_best_of_selects_final_only_on_strict_initial_lift() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let evaluation = evaluation_with(-0.49, 1.5, -0.3, 1.4);
        let initial_identity = evaluation.initial.state_identity;
        let final_identity = evaluation.final_candidate.state_identity;
        let publication = build_critical_gpu_alpha_publication_with_clock(
            &[(-1.017_f32, 2.0_f32)],
            &plan(0, deadline),
            evaluation,
            || just_before(deadline),
        )
        .expect("strict certified post-Adam lift");
        let tag = publication.pair.transport_tag.expect("tagged pair");
        assert_eq!(tag.selected_pair, CriticalGpuAlphaSelectedPair::Final);
        assert_eq!(tag.initial_lower, -0.49);
        assert_eq!(tag.final_lower, -0.3);
        assert_eq!(tag.merged_lower, -0.3);
        assert_ne!(initial_identity, final_identity);
        assert_eq!(tag.state_identity, final_identity);
        assert_eq!(
            alpha_state_identity(&publication.pair.alpha_state),
            Some(final_identity),
            "the final direct-C bound must transport only its exact final state"
        );
        assert_eq!(publication.pair.objective_bounds[0].0, -0.3);
    }

    #[test]
    fn critical_gpu_alpha_bracket_publication_retains_selected_provenance() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let publication = build_critical_gpu_alpha_publication_with_clock(
            &[(-1.017_f32, 2.0_f32)],
            &plan(0, deadline),
            bracket_evaluation(-0.49, -0.3),
            || just_before(deadline),
        )
        .expect("strict certified bracket lift");
        let tag = publication.pair.transport_tag.expect("tagged pair");
        let bracket = tag.bracket.expect("bracket provenance");
        assert_eq!(tag.selected_pair, CriticalGpuAlphaSelectedPair::Final);
        assert_eq!(bracket.selected_ordinal, 1);
        assert_eq!(bracket.selected_lr.to_bits(), 0.1_f32.to_bits());
        assert_eq!(bracket.evaluated_candidates, 3);
        assert_eq!(bracket.gradient_replays, 1);
        assert_eq!(
            alpha_state_identity(&publication.pair.alpha_state),
            Some(tag.state_identity)
        );

        let regression = bracket_evaluation(-0.49, -0.9);
        let initial_identity = regression.initial.state_identity;
        let publication = build_critical_gpu_alpha_publication_with_clock(
            &[(-1.017_f32, 2.0_f32)],
            &plan(0, deadline),
            regression,
            || just_before(deadline),
        )
        .expect("initial certified pair remains a publishable historical lift");
        let tag = publication.pair.transport_tag.expect("regression tag");
        assert_eq!(tag.selected_pair, CriticalGpuAlphaSelectedPair::Initial);
        assert_eq!(tag.state_identity, initial_identity);
        assert_eq!(
            alpha_state_identity(&publication.pair.alpha_state),
            Some(initial_identity),
            "a regressive bracket must transport the initial bound's exact state"
        );
        assert_eq!(
            tag.bracket.expect("search provenance").selected_ordinal,
            1,
            "search provenance describes its best candidate even when root retains initial"
        );

        let mut mismatched = bracket_evaluation(-0.49, -0.3);
        mismatched
            .search_provenance
            .as_mut()
            .expect("provenance")
            .selected_ordinal = 0;
        assert_eq!(
            build_critical_gpu_alpha_publication_with_clock(
                &[(-1.017_f32, 2.0_f32)],
                &plan(0, deadline),
                mismatched,
                || just_before(deadline),
            )
            .err(),
            Some(CriticalGpuAlphaRefusal::SearchProvenanceMismatch)
        );
    }

    #[test]
    fn critical_gpu_alpha_publication_rejects_late_invalid_disjoint_and_mismatched_pairs() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let plan = plan(0, deadline);
        let historical = vec![(-1.0_f32, 2.0_f32)];
        let original_bits = (historical[0].0.to_bits(), historical[0].1.to_bits());
        let before = || just_before(deadline);

        assert_eq!(
            build_critical_gpu_alpha_publication_with_clock(
                &historical,
                &plan,
                evaluation(-2.0, 1.5),
                before,
            )
            .err(),
            Some(CriticalGpuAlphaRefusal::NoCertifiedImprovement)
        );
        assert_eq!(
            build_critical_gpu_alpha_publication_with_clock(
                &historical,
                &plan,
                evaluation(3.0, 4.0),
                before,
            )
            .err(),
            Some(CriticalGpuAlphaRefusal::CandidateDisjoint)
        );
        assert_eq!(
            build_critical_gpu_alpha_publication_with_clock(
                &historical,
                &plan,
                evaluation(f32::NEG_INFINITY, 1.0),
                before,
            )
            .err(),
            Some(CriticalGpuAlphaRefusal::InvalidFinalCandidate)
        );
        assert_eq!(
            build_critical_gpu_alpha_publication_with_clock(
                &historical,
                &plan,
                evaluation(-0.25, 1.5),
                || deadline,
            )
            .err(),
            Some(CriticalGpuAlphaRefusal::CompletedAfterDeadline)
        );

        let mut mismatched = evaluation(-0.25, 1.5);
        mismatched.final_candidate.state_identity.fingerprint ^= 1;
        assert_eq!(
            build_critical_gpu_alpha_publication_with_clock(
                &historical,
                &plan,
                mismatched,
                before,
            )
            .err(),
            Some(CriticalGpuAlphaRefusal::StateIdentityMismatch)
        );
        assert_eq!(
            (historical[0].0.to_bits(), historical[0].1.to_bits()),
            original_bits,
            "all refusals leave the source pair untouched"
        );

        let calls = Cell::new(0);
        let late_final_publication_sample = || {
            let call = calls.get();
            calls.set(call + 1);
            if call < 2 {
                just_before(deadline)
            } else {
                deadline
            }
        };
        assert_eq!(
            build_critical_gpu_alpha_publication_with_clock(
                &historical,
                &plan,
                evaluation(-0.25, 1.5),
                late_final_publication_sample,
            )
            .err(),
            Some(CriticalGpuAlphaRefusal::CompletedAfterDeadline),
            "publication must recheck after constructing the replacement"
        );
    }

    #[test]
    fn critical_gpu_alpha_tag_survives_root_child_and_dense_shim() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let publication = build_critical_gpu_alpha_publication_with_clock(
            &[(-1.0_f32, 2.0_f32)],
            &plan(0, deadline),
            evaluation(-0.25, 1.5),
            || just_before(deadline),
        )
        .expect("test publication");
        let tag = publication.pair.transport_tag.expect("tag");

        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("input box");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
        graph.set_output("relu0");
        let mut root = MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            publication.pair.objective_bounds,
            &input,
            &[0.0],
            false,
        )
        .expect("root domain");
        root.set_alpha_state(publication.pair.alpha_state);
        root.set_cached_las(vec![Some(CachedLinearBounds::default())])
            .expect("one-row cache");
        root.clear_cached_la_for_objective(tag.row_index);
        assert!(root.cached_la_for_objective(tag.row_index).is_none());
        assert_eq!(
            alpha_state_identity(root.alpha_state()),
            Some(tag.state_identity)
        );

        let child = root
            .with_constraint(
                &graph,
                GraphNeuronConstraint::new("relu0".into(), 0, true, 1.0)
                    .expect("finite constraint"),
                false,
                &[0.0],
            )
            .expect("child creation")
            .expect("feasible child");
        assert_eq!(
            child.alpha_state().alpha("relu0", 1).to_bits(),
            0.75_f32.to_bits()
        );
        let shim = super::super::batched::graph_bab_domain_shim(&child);
        assert_eq!(
            shim.alpha_state.alpha("relu0", 1).to_bits(),
            0.75_f32.to_bits()
        );
    }

    #[test]
    fn critical_gpu_alpha_telemetry_is_stable_and_direct() {
        assert_eq!(
            critical_gpu_alpha_telemetry_line_if(
                false,
                CriticalGpuAlphaTelemetry::BackendSelected {
                    backend: CriticalGpuSpecBackend::Factory,
                    row: 3,
                },
            ),
            None
        );
        assert_eq!(
            critical_gpu_alpha_telemetry_line_if(
                true,
                CriticalGpuAlphaTelemetry::BackendSelected {
                    backend: CriticalGpuSpecBackend::Factory,
                    row: 3,
                },
            )
            .as_deref(),
            Some("[critical-gpu-alpha] status=selected backend=factory row=3")
        );

        let deadline = Instant::now() + Duration::from_secs(1);
        let legacy = build_critical_gpu_alpha_publication_with_clock(
            &[(-1.0_f32, 2.0_f32)],
            &plan(0, deadline),
            evaluation(-0.25, 1.5),
            || just_before(deadline),
        )
        .expect("legacy publication");
        let legacy_tag = legacy.pair.transport_tag.as_ref().expect("legacy tag");
        let legacy_line = critical_gpu_alpha_telemetry_line_if(
            true,
            CriticalGpuAlphaTelemetry::Accepted {
                backend: CriticalGpuSpecBackend::Local,
                tag: legacy_tag,
            },
        )
        .expect("legacy telemetry");
        assert_eq!(
            legacy_line,
            format!(
                "[critical-gpu-alpha] status=accepted backend=local row=0 \
                 initial_lower=-1.25 final_lower=-0.25 selected=final historical_lower=-1 \
                 merged_lower=-0.25 lift=0.75 alpha_params=2 alpha_fingerprint={:016x} \
                 cache=invalidated",
                legacy_tag.state_identity.fingerprint
            ),
            "bracket-off retains the sealed accepted line exactly"
        );

        let bracket_sample = bracket_evaluation(-0.49, -0.3);
        let candidate = &bracket_sample
            .search_provenance
            .as_ref()
            .expect("bracket provenance")
            .candidates[1];
        assert_eq!(
            critical_gpu_alpha_telemetry_line_if(
                true,
                CriticalGpuAlphaTelemetry::BracketCandidate { row: 95, candidate },
            )
            .expect("candidate telemetry"),
            format!(
                "[critical-gpu-alpha-candidate] row=95 ordinal=1 t=1 lr=0.1 lower=-0.3 \
                 lift=0.19 alpha_params=2 fp={:016x}",
                candidate.state_identity.fingerprint
            )
        );

        let bracket = build_critical_gpu_alpha_publication_with_clock(
            &[(-1.0_f32, 2.0_f32)],
            &plan(0, deadline),
            bracket_evaluation(-0.49, -0.3),
            || just_before(deadline),
        )
        .expect("bracket publication");
        let bracket_tag = bracket.pair.transport_tag.as_ref().expect("bracket tag");
        let bracket_line = critical_gpu_alpha_telemetry_line_if(
            true,
            CriticalGpuAlphaTelemetry::Accepted {
                backend: CriticalGpuSpecBackend::Local,
                tag: bracket_tag,
            },
        )
        .expect("bracket telemetry");
        assert!(
            bracket_line.contains("selected_ordinal=1 selected_lr=0.1 candidates=3 replays=1"),
            "accepted telemetry must report the provenance used by selection: {bracket_line}"
        );
    }
}

#[cfg(test)]
mod active_set_gpu_alpha_root_tests {
    use super::{
        active_set_gpu_alpha_telemetry_line_if, active_set_scalar_cascade_telemetry_line_if,
        build_active_set_gpu_alpha_publication_with_clock, build_active_set_scalar_cascade_plan,
        build_active_set_scalar_cascade_publication_with_clock,
        build_critical_gpu_alpha_publication_with_clock,
        classify_active_set_gpu_alpha_factory_error, classify_active_set_scalar_cascade_survivor,
        classify_complete_active_set_gpu_alpha, invalidate_active_set_stale_transport,
        objective_bounds_fingerprint, retain_active_set_pair_on_cascade_refusal,
        root_critical_gpu_alpha_active_set_cascade_enabled_from_value,
        root_critical_gpu_alpha_active_set_cascade_enabled_if,
        root_critical_gpu_alpha_active_set_enabled_from_value,
        root_critical_gpu_alpha_active_set_enabled_if, ActiveSetGpuAlphaRootRefusal,
        ActiveSetGpuAlphaTelemetry, ActiveSetScalarCascadeRefusal, ActiveSetScalarCascadeTelemetry,
        CriticalGpuAlphaRefusal, CriticalGpuSpecBackend, CriticalGpuSpecPlan,
    };
    use crate::batched_domain::CachedLinearBounds;
    use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
    use crate::beta_crown::engine::graph::multi_objective::active_set_gpu_alpha::{
        complete_initial_execution_output_for_test, ActiveSetGpuAlphaCertifiedPair,
        ActiveSetGpuAlphaClassification, ActiveSetGpuAlphaExecutionRefusal, ActiveSetGpuAlphaPlan,
        ActiveSetGpuAlphaRefusal,
    };
    use crate::beta_crown::engine::graph::multi_objective::critical_gpu_alpha::{
        alpha_state_identity, CriticalGpuAlphaCertifiedPair, CriticalGpuAlphaStepOutput,
        CriticalGpuAlphaStepRefusal,
    };
    use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};
    use ndarray::{arr1, arr2};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    fn active_plan(historical: &[(f32, f32)], thresholds: &[f32]) -> ActiveSetGpuAlphaPlan {
        match classify_complete_active_set_gpu_alpha(historical, thresholds)
            .expect("valid active classification")
        {
            ActiveSetGpuAlphaClassification::Optimize(plan) => plan,
            ActiveSetGpuAlphaClassification::DelegateSealedCriticalRow(_) => {
                panic!("fixture must retain at least two active rows")
            }
        }
    }

    #[test]
    fn active_set_factory_error_preserves_deadline_refusal() {
        assert_eq!(
            classify_active_set_gpu_alpha_factory_error(&NyError::DeadlineExceeded(
                "test deadline".into()
            )),
            ActiveSetGpuAlphaRootRefusal::Execution(ActiveSetGpuAlphaExecutionRefusal::Step(
                CriticalGpuAlphaStepRefusal::DeadlineExpired
            ))
        );
        assert_eq!(
            classify_active_set_gpu_alpha_factory_error(&NyError::InvalidSpec(
                "test failure".into()
            )),
            ActiveSetGpuAlphaRootRefusal::FactoryError
        );
    }

    fn state(first: f32, second: f32) -> GraphDomainAlphaState {
        let mut state = GraphDomainAlphaState::empty();
        state.insert("relu0".into(), 0, AlphaNeuronState::new(first));
        state.insert("relu0".into(), 1, AlphaNeuronState::new(second));
        state
    }

    fn vector_bounds(rows: &[(f32, f32)]) -> BoundedTensor {
        BoundedTensor::new_allow_infinite(
            arr1(&rows.iter().map(|row| row.0).collect::<Vec<_>>()).into_dyn(),
            arr1(&rows.iter().map(|row| row.1).collect::<Vec<_>>()).into_dyn(),
        )
        .expect("finite vector enclosure")
    }

    fn complete_output(
        plan: &ActiveSetGpuAlphaPlan,
        rows: &[(f32, f32)],
    ) -> crate::beta_crown::engine::graph::multi_objective::active_set_gpu_alpha::
    ActiveSetGpuAlphaExecutionOutput{
        let pair =
            ActiveSetGpuAlphaCertifiedPair::new(plan, vector_bounds(rows), state(0.25, 0.75))
                .expect("valid whole pair");
        complete_initial_execution_output_for_test(pair, 0.1)
    }

    fn instant_before(deadline: Instant, duration: Duration) -> Instant {
        deadline
            .checked_sub(duration)
            .expect("test deadline offset must be representable")
    }

    fn active_pair_with_rows(rows: &[(f32, f32)]) -> super::CriticalGpuAlphaRootPair {
        let historical = vec![(-2.0_f32, 2.0), (-1.0, 2.5)];
        let thresholds = vec![0.0_f32, 0.0];
        let plan = active_plan(&historical, &thresholds);
        let deadline = Instant::now() + Duration::from_secs(2);
        build_active_set_gpu_alpha_publication_with_clock(
            &historical,
            &plan,
            complete_output(&plan, rows),
            deadline,
            || instant_before(deadline, Duration::from_secs(1)),
        )
        .expect("active fixture must publish")
    }

    fn scalar_pair(
        lower: f32,
        upper: f32,
        first: f32,
        second: f32,
    ) -> CriticalGpuAlphaCertifiedPair {
        let state = state(first, second);
        let state_identity = alpha_state_identity(&state).expect("valid scalar alpha state");
        CriticalGpuAlphaCertifiedPair {
            bounds: vector_bounds(&[(lower, upper)]),
            state,
            state_identity,
        }
    }

    fn scalar_evaluation(initial_lower: f32, final_lower: f32) -> CriticalGpuAlphaStepOutput {
        CriticalGpuAlphaStepOutput {
            initial: scalar_pair(initial_lower, 1.75, 0.2, 0.6),
            final_candidate: scalar_pair(final_lower, 1.5, 0.4, 0.8),
            search_provenance: None,
        }
    }

    fn cascade_plan(
        active_pair: &super::CriticalGpuAlphaRootPair,
        deadline: Instant,
    ) -> CriticalGpuSpecPlan {
        build_active_set_scalar_cascade_plan(
            &active_pair.objective_bounds,
            &arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]),
            &[0.0, 0.0],
            deadline,
            instant_before(deadline, Duration::from_secs(1)),
        )
        .expect("fixture has one cascade survivor")
    }

    fn pair_snapshot(
        pair: &super::CriticalGpuAlphaRootPair,
    ) -> (
        Vec<(u32, u32)>,
        super::ActiveSetGpuAlphaTransportTag,
        super::ActiveSetGpuAlphaFullStateIdentity,
    ) {
        (
            pair.objective_bounds
                .iter()
                .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
                .collect(),
            pair.active_set_transport_tag.clone().expect("active tag"),
            super::active_set_full_state_identity(&pair.alpha_state).expect("active state"),
        )
    }

    #[test]
    fn active_set_gate_is_exact_and_not_read_outside_bracket() {
        for malformed in [None, Some("0"), Some("true"), Some(" 1"), Some("1 ")] {
            assert!(!root_critical_gpu_alpha_active_set_enabled_from_value(
                malformed
            ));
        }
        assert!(root_critical_gpu_alpha_active_set_enabled_from_value(Some(
            "1"
        )));

        let reads = Cell::new(0);
        assert!(!root_critical_gpu_alpha_active_set_enabled_if(
            false,
            || {
                reads.set(reads.get() + 1);
                Some("1".to_string())
            }
        ));
        assert_eq!(reads.get(), 0, "bracket-off must perform zero env reads");
        assert!(root_critical_gpu_alpha_active_set_enabled_if(true, || {
            reads.set(reads.get() + 1);
            Some("1".to_string())
        }));
        assert_eq!(reads.get(), 1);
    }

    #[test]
    fn cascade_gate_is_exact_and_not_read_outside_parent_active_set() {
        for malformed in [None, Some("0"), Some("true"), Some(" 1"), Some("1 ")] {
            assert!(!root_critical_gpu_alpha_active_set_cascade_enabled_from_value(malformed));
        }
        assert!(root_critical_gpu_alpha_active_set_cascade_enabled_from_value(Some("1")));

        let reads = Cell::new(0);
        assert!(!root_critical_gpu_alpha_active_set_cascade_enabled_if(
            false,
            || {
                reads.set(reads.get() + 1);
                Some("1".to_string())
            }
        ));
        assert_eq!(
            reads.get(),
            0,
            "parent active-set off must perform zero cascade env reads"
        );
        assert!(root_critical_gpu_alpha_active_set_cascade_enabled_if(
            true,
            || {
                reads.set(reads.get() + 1);
                Some("1".to_string())
            }
        ));
        assert_eq!(reads.get(), 1);
    }

    #[test]
    fn complete_classification_keeps_equality_active_and_k1_delegates() {
        let delegated =
            classify_complete_active_set_gpu_alpha(&[(0.0_f32, 1.0), (1.0, 2.0)], &[0.0, 0.0])
                .expect("equality row remains the sole active row");
        match delegated {
            ActiveSetGpuAlphaClassification::DelegateSealedCriticalRow(row) => {
                assert_eq!(row.source_row_index(), 0);
            }
            ActiveSetGpuAlphaClassification::Optimize(_) => {
                panic!("K=1 must delegate to the sealed scalar route")
            }
        }
    }

    #[test]
    fn k_above_eight_refuses_before_any_execution_seam() {
        let historical = vec![(f32::NAN, f32::NEG_INFINITY); 9];
        let thresholds = vec![0.0_f32; 9];
        let execution_calls = Cell::new(0);
        let refusal = classify_complete_active_set_gpu_alpha(&historical, &thresholds)
            .expect_err("K=9 must refuse by count before validating poison rows");
        assert!(matches!(
            refusal,
            ActiveSetGpuAlphaRootRefusal::Classification(
                ActiveSetGpuAlphaRefusal::TooManyUnresolvedRows {
                    count: 9,
                    maximum: 8
                }
            )
        ));
        if refusal.telemetry_reason() != "too_many_unresolved_rows" {
            execution_calls.set(execution_calls.get() + 1);
        }
        assert_eq!(execution_calls.get(), 0);
        assert_eq!(
            active_set_gpu_alpha_telemetry_line_if(
                true,
                ActiveSetGpuAlphaTelemetry::PlanRefused {
                    rows: Some(9),
                    reason: refusal,
                },
            )
            .expect("direct refusal telemetry"),
            "[active-set-gpu-alpha] status=refused backend=unselected rows=9 \
             reason=too_many_unresolved_rows"
        );
    }

    #[test]
    fn publication_intersects_one_whole_vector_and_transports_its_exact_state() {
        let historical = vec![(-2.0_f32, 2.0), (1.0, 3.0), (-1.0, 2.5)];
        let thresholds = vec![0.0_f32, 0.0, 0.0];
        let plan = active_plan(&historical, &thresholds);
        assert_eq!(
            plan.rows()
                .iter()
                .map(|row| row.source_row_index())
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        let deadline = Instant::now() + Duration::from_secs(1);
        let pair = build_active_set_gpu_alpha_publication_with_clock(
            &historical,
            &plan,
            complete_output(&plan, &[(-0.5, 1.5), (0.25, 2.0)]),
            deadline,
            || instant_before(deadline, Duration::from_nanos(1)),
        )
        .expect("overlapping strict vector lift must publish");
        let tag = pair
            .active_set_transport_tag
            .as_ref()
            .expect("active publication tag");
        assert_eq!(tag.rows.len(), 2);
        assert_eq!(tag.rows[0].source_row_index, 0);
        assert_eq!(tag.rows[0].candidate_lower.to_bits(), (-0.5_f32).to_bits());
        assert_eq!(tag.rows[1].source_row_index, 2);
        assert_eq!(tag.rows[1].candidate_lower.to_bits(), 0.25_f32.to_bits());
        assert_eq!(tag.gradient_replays, 1);
        assert_eq!(tag.evaluated_candidates, 3);
        assert_eq!(
            pair.objective_bounds,
            vec![(-0.5, 2.0), (1.0, 3.0), (0.25, 2.5)]
        );
        assert_eq!(
            super::active_set_full_state_identity(&pair.alpha_state),
            Some(tag.state_identity)
        );
    }

    #[test]
    fn accepted_k2_to_k1_cascade_intersects_certificates_and_transports_scalar_state() {
        let active_pair = active_pair_with_rows(&[(0.25, 1.5), (-0.5, 2.0)]);
        let active_identity =
            super::active_set_full_state_identity(&active_pair.alpha_state).expect("active state");
        assert_eq!(
            classify_active_set_scalar_cascade_survivor(&active_pair.objective_bounds, &[0.0, 0.0],),
            Ok(1),
            "the accepted K=2 vector must leave exactly row 1"
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        let plan = cascade_plan(&active_pair, deadline);
        assert_eq!(plan.row_index, 1);
        assert_eq!(
            plan.deadline, deadline,
            "cascade must reuse the original authority deadline exactly"
        );
        let scalar_publication = build_critical_gpu_alpha_publication_with_clock(
            &active_pair.objective_bounds,
            &plan,
            scalar_evaluation(-0.4, 0.3),
            || instant_before(deadline, Duration::from_secs(1)),
        )
        .expect("scalar survivor has a strict certified lift");
        let expected_scalar_identity = scalar_publication
            .pair
            .transport_tag
            .as_ref()
            .expect("scalar tag")
            .state_identity;
        let pair = build_active_set_scalar_cascade_publication_with_clock(
            &active_pair,
            scalar_publication,
            &plan,
            &[0.0, 0.0],
            deadline,
            || instant_before(deadline, Duration::from_millis(1)),
        )
        .expect("two sound enclosures may be intersected across alpha states");

        assert_eq!(pair.objective_bounds, vec![(0.25, 2.0), (0.3, 2.5)]);
        assert!(
            pair.active_set_transport_tag.is_none(),
            "the final state must not masquerade as the active-set producing state"
        );
        let scalar_tag = pair.transport_tag.as_ref().expect("scalar provenance");
        let cascade_tag = pair
            .active_set_scalar_cascade_transport_tag
            .as_ref()
            .expect("cross-state intersection provenance");
        assert_eq!(cascade_tag.survivor_row, 1);
        assert_eq!(cascade_tag.active_set.state_identity, active_identity);
        assert_eq!(cascade_tag.final_state_identity, expected_scalar_identity);
        assert_eq!(scalar_tag.state_identity, expected_scalar_identity);
        assert_ne!(
            cascade_tag.active_set.state_identity.fingerprint(),
            cascade_tag.final_state_identity.fingerprint,
            "the tag must retain both distinct producing-state identities"
        );
        assert_eq!(
            alpha_state_identity(&pair.alpha_state),
            Some(expected_scalar_identity)
        );
        assert_eq!(
            objective_bounds_fingerprint(&pair.objective_bounds),
            Some(cascade_tag.published_bounds_fingerprint)
        );

        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input box");
        let mut root = MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            pair.objective_bounds,
            &input,
            &[0.0, 0.0],
            false,
        )
        .expect("root");
        root.set_alpha_state(pair.alpha_state);
        root.set_cached_las(vec![
            Some(CachedLinearBounds::default()),
            Some(CachedLinearBounds::default()),
        ])
        .expect("two aligned caches");
        root.set_per_disjunct_alphas(vec![state(0.1, 0.2); 2]);
        invalidate_active_set_stale_transport(&mut root);
        assert!(root.cached_las().iter().all(Option::is_none));
        assert!(root.per_disjunct_alphas().is_none());
        assert_eq!(
            alpha_state_identity(root.alpha_state()),
            Some(expected_scalar_identity),
            "cache invalidation must not mutate the transported scalar state"
        );
    }

    #[test]
    fn cascade_no_survivor_and_multiple_survivors_refuse_before_scalar_execution() {
        let calls = Cell::new(0);
        let deadline = Instant::now() + Duration::from_secs(1);
        let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0_f32, 0.0];
        for (bounds, expected) in [
            (
                vec![(0.1_f32, 1.0), (0.2, 1.0)],
                ActiveSetScalarCascadeRefusal::NoSurvivor,
            ),
            (
                vec![(-0.1_f32, 1.0), (-0.2, 1.0)],
                ActiveSetScalarCascadeRefusal::TooManySurvivors { count: 2 },
            ),
        ] {
            let plan = build_active_set_scalar_cascade_plan(
                &bounds,
                &spec,
                &thresholds,
                deadline,
                instant_before(deadline, Duration::from_millis(1)),
            );
            if plan.is_ok() {
                calls.set(calls.get() + 1);
            }
            assert_eq!(plan.err(), Some(expected));
        }
        assert_eq!(calls.get(), 0);
        assert_eq!(
            classify_active_set_scalar_cascade_survivor(&[(0.0_f32, 1.0), (0.1, 1.0)], &thresholds,),
            Ok(0),
            "strict root semantics keep threshold equality unresolved"
        );
    }

    #[test]
    fn cascade_deadline_engine_refusal_and_regression_retain_active_pair_bit_exactly() {
        // Final composition deadline refusal.
        let active_pair = active_pair_with_rows(&[(0.25, 1.5), (-0.5, 2.0)]);
        let before = pair_snapshot(&active_pair);
        let deadline = Instant::now() + Duration::from_secs(2);
        let plan = cascade_plan(&active_pair, deadline);
        let scalar_publication = build_critical_gpu_alpha_publication_with_clock(
            &active_pair.objective_bounds,
            &plan,
            scalar_evaluation(-0.4, 0.3),
            || instant_before(deadline, Duration::from_secs(1)),
        )
        .expect("scalar publication before final cascade clock");
        let cascade = build_active_set_scalar_cascade_publication_with_clock(
            &active_pair,
            scalar_publication,
            &plan,
            &[0.0, 0.0],
            deadline,
            || deadline,
        );
        assert_eq!(
            cascade.as_ref().err(),
            Some(&ActiveSetScalarCascadeRefusal::CompletedAfterDeadline)
        );
        let (retained, reason) = retain_active_set_pair_on_cascade_refusal(active_pair, cascade);
        assert_eq!(
            reason,
            Some(ActiveSetScalarCascadeRefusal::CompletedAfterDeadline)
        );
        assert_eq!(pair_snapshot(&retained), before);
        assert!(retained.transport_tag.is_none());
        assert!(retained.active_set_scalar_cascade_transport_tag.is_none());

        // Engine/candidate refusal.
        let active_pair = active_pair_with_rows(&[(0.25, 1.5), (-0.5, 2.0)]);
        let before = pair_snapshot(&active_pair);
        let engine_refusal = ActiveSetScalarCascadeRefusal::Scalar(CriticalGpuAlphaRefusal::Step(
            CriticalGpuAlphaStepRefusal::NoSoundGpuRoute,
        ));
        let (retained, reason) =
            retain_active_set_pair_on_cascade_refusal(active_pair, Err(engine_refusal));
        assert_eq!(reason, Some(engine_refusal));
        assert_eq!(pair_snapshot(&retained), before);

        // A scalar bracket whose best certified pair does not beat the active
        // survivor is a regression relative to the active authority.
        let active_pair = active_pair_with_rows(&[(0.25, 1.5), (-0.5, 2.0)]);
        let before = pair_snapshot(&active_pair);
        let deadline = Instant::now() + Duration::from_secs(2);
        let plan = cascade_plan(&active_pair, deadline);
        let regression = build_critical_gpu_alpha_publication_with_clock(
            &active_pair.objective_bounds,
            &plan,
            scalar_evaluation(-1.5, -1.0),
            || instant_before(deadline, Duration::from_secs(1)),
        )
        .map(|_| unreachable!("regressive scalar must not publish"))
        .map_err(ActiveSetScalarCascadeRefusal::Scalar);
        assert_eq!(
            regression.as_ref().err(),
            Some(&ActiveSetScalarCascadeRefusal::Scalar(
                CriticalGpuAlphaRefusal::NoCertifiedImprovement,
            ))
        );
        let (retained, reason) = retain_active_set_pair_on_cascade_refusal(active_pair, regression);
        assert_eq!(
            reason,
            Some(ActiveSetScalarCascadeRefusal::Scalar(
                CriticalGpuAlphaRefusal::NoCertifiedImprovement,
            ))
        );
        assert_eq!(pair_snapshot(&retained), before);
    }

    #[test]
    fn cascade_rejects_unproved_non_survivor_transplant_and_retains_active_pair() {
        let active_pair = active_pair_with_rows(&[(0.25, 1.5), (-0.5, 2.0)]);
        let before = pair_snapshot(&active_pair);
        let deadline = Instant::now() + Duration::from_secs(2);
        let plan = cascade_plan(&active_pair, deadline);
        let mut scalar_publication = build_critical_gpu_alpha_publication_with_clock(
            &active_pair.objective_bounds,
            &plan,
            scalar_evaluation(-0.4, 0.3),
            || instant_before(deadline, Duration::from_secs(1)),
        )
        .expect("valid scalar publication");
        scalar_publication.pair.objective_bounds[0].0 = 0.5;
        let cascade = build_active_set_scalar_cascade_publication_with_clock(
            &active_pair,
            scalar_publication,
            &plan,
            &[0.0, 0.0],
            deadline,
            || instant_before(deadline, Duration::from_millis(1)),
        );
        assert_eq!(
            cascade.as_ref().err(),
            Some(&ActiveSetScalarCascadeRefusal::CertifiedIntersectionMismatch)
        );
        let (retained, reason) = retain_active_set_pair_on_cascade_refusal(active_pair, cascade);
        assert_eq!(
            reason,
            Some(ActiveSetScalarCascadeRefusal::CertifiedIntersectionMismatch)
        );
        assert_eq!(pair_snapshot(&retained), before);
    }

    #[test]
    fn late_active_publication_refuses_without_mutating_history() {
        let historical = vec![(-2.0_f32, 2.0), (-1.0, 2.5)];
        let thresholds = vec![0.0_f32, 0.0];
        let plan = active_plan(&historical, &thresholds);
        let before = historical.clone();
        let deadline = Instant::now() + Duration::from_secs(1);
        let refusal = build_active_set_gpu_alpha_publication_with_clock(
            &historical,
            &plan,
            complete_output(&plan, &[(-0.5, 1.5), (0.25, 2.0)]),
            deadline,
            || deadline,
        )
        .expect_err("publication at the authority boundary must refuse");
        assert_eq!(
            refusal,
            ActiveSetGpuAlphaRootRefusal::CompletedAfterDeadline
        );
        assert_eq!(historical, before);
    }

    #[test]
    fn accepted_active_state_clears_every_cache_and_per_disjunct_transport() {
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input box");
        let mut root = MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            vec![(-1.0, 2.0), (-2.0, 3.0), (-3.0, 4.0)],
            &input,
            &[0.0, 0.0, 0.0],
            false,
        )
        .expect("root");
        root.set_cached_las(vec![
            Some(CachedLinearBounds::default()),
            Some(CachedLinearBounds::default()),
            Some(CachedLinearBounds::default()),
        ])
        .expect("three aligned caches");
        root.set_per_disjunct_alphas(vec![state(0.1, 0.2); 3]);

        invalidate_active_set_stale_transport(&mut root);
        assert!(root.cached_las().iter().all(Option::is_none));
        assert!(root.per_disjunct_alphas().is_none());
    }

    #[test]
    fn accepted_telemetry_binds_backend_vector_and_full_state_fingerprint() {
        let historical = vec![(-2.0_f32, 2.0), (-1.0, 2.5)];
        let thresholds = vec![0.0_f32, 0.0];
        let plan = active_plan(&historical, &thresholds);
        let deadline = Instant::now() + Duration::from_secs(1);
        let pair = build_active_set_gpu_alpha_publication_with_clock(
            &historical,
            &plan,
            complete_output(&plan, &[(-0.5, 1.5), (0.25, 2.0)]),
            deadline,
            || instant_before(deadline, Duration::from_nanos(1)),
        )
        .expect("publication");
        let tag = pair.active_set_transport_tag.as_ref().expect("tag");
        let line = active_set_gpu_alpha_telemetry_line_if(
            true,
            ActiveSetGpuAlphaTelemetry::Accepted {
                backend: CriticalGpuSpecBackend::Local,
                tag,
            },
        )
        .expect("telemetry");
        assert!(line.contains("status=accepted backend=local rows=2 source_rows=0,1"));
        assert!(line.contains("candidates=3 replays=1"));
        assert!(line.contains(&format!(
            "state_fp={:016x}",
            tag.state_identity.fingerprint()
        )));
        assert!(line.contains(&format!("pair_fp={:016x}", tag.pair_fingerprint.value())));
        assert!(line.ends_with("cache=all_invalidated"));
    }

    #[test]
    fn cascade_telemetry_distinguishes_skip_refusal_and_certified_intersection() {
        assert_eq!(
            active_set_scalar_cascade_telemetry_line_if(
                true,
                ActiveSetScalarCascadeTelemetry::Refused {
                    backend: None,
                    row: None,
                    reason: ActiveSetScalarCascadeRefusal::NoSurvivor,
                },
            )
            .as_deref(),
            Some(
                "[active-set-gpu-alpha-cascade] status=refused backend=unselected \
                 row=none survivors=0 reason=no_survivor"
            )
        );
        assert_eq!(
            active_set_scalar_cascade_telemetry_line_if(
                true,
                ActiveSetScalarCascadeTelemetry::Refused {
                    backend: Some(CriticalGpuSpecBackend::Local),
                    row: Some(46),
                    reason: ActiveSetScalarCascadeRefusal::TooManySurvivors { count: 2 },
                },
            )
            .as_deref(),
            Some(
                "[active-set-gpu-alpha-cascade] status=refused backend=local \
                 row=46 survivors=2 reason=too_many_survivors"
            )
        );

        let active_pair = active_pair_with_rows(&[(0.25, 1.5), (-0.5, 2.0)]);
        let deadline = Instant::now() + Duration::from_secs(2);
        let plan = cascade_plan(&active_pair, deadline);
        let scalar_publication = build_critical_gpu_alpha_publication_with_clock(
            &active_pair.objective_bounds,
            &plan,
            scalar_evaluation(-0.4, 0.3),
            || instant_before(deadline, Duration::from_secs(1)),
        )
        .expect("scalar publication");
        let pair = build_active_set_scalar_cascade_publication_with_clock(
            &active_pair,
            scalar_publication,
            &plan,
            &[0.0, 0.0],
            deadline,
            || instant_before(deadline, Duration::from_millis(1)),
        )
        .expect("cascade publication");
        let line = active_set_scalar_cascade_telemetry_line_if(
            true,
            ActiveSetScalarCascadeTelemetry::Accepted {
                backend: CriticalGpuSpecBackend::Local,
                tag: pair
                    .active_set_scalar_cascade_transport_tag
                    .as_ref()
                    .expect("cascade tag"),
                scalar: pair.transport_tag.as_ref().expect("scalar tag"),
            },
        )
        .expect("accepted telemetry");
        assert!(line.contains("status=accepted backend=local row=1"));
        assert!(line.contains("historical_lower=-0.5 merged_lower=0.3 lift=0.8"));
        assert!(line.contains("active_rows=2"));
        assert!(line.ends_with("cache=all_invalidated"));
    }
}

#[cfg(test)]
mod root_spec_prune_tests {
    use super::{
        build_atomic_root_c_compact_rows, build_root_spec_prune_plan,
        compact_atomic_root_c_reconstruction_succeeded, expand_root_spec_cache,
        finalize_root_spec_prune_publication, map_compact_atomic_root_c_row,
        map_compact_atomic_root_c_rows, merge_root_spec_pruned_bounds,
        publish_pending_atomic_root_alpha, retry_full_spec_after_compact_merge_failure,
        root_prebound_certifies,
    };
    use crate::batched_domain::CachedLinearBounds;
    use crate::bounds::GraphAlphaState;
    use ndarray::{arr1, arr2};
    use ny_tensor::BoundedTensor;

    fn fixture() -> (BoundedTensor, ndarray::Array2<f32>, Vec<Vec<f32>>, Vec<f32>) {
        let output = BoundedTensor::new(
            arr1(&[1.0_f32, 2.0]).into_dyn(),
            arr1(&[3.0_f32, 4.0]).into_dyn(),
        )
        .unwrap();
        let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [-1.0, 0.0]]);
        let objectives = spec.outer_iter().map(|row| row.to_vec()).collect();
        // Rows 0 and 2 are certified by the output box; row 1 remains active.
        let thresholds = vec![0.0_f32, 2.5, -4.0];
        (output, spec, objectives, thresholds)
    }

    fn noncontiguous_fixture() -> (BoundedTensor, ndarray::Array2<f32>, Vec<Vec<f32>>, Vec<f32>) {
        let output = BoundedTensor::new(
            arr1(&[2.0_f32, -2.0, 4.0, -4.0]).into_dyn(),
            arr1(&[3.0_f32, 1.0, 5.0, 2.0]).into_dyn(),
        )
        .expect("valid output box");
        let spec = arr2(&[
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let objectives = spec.outer_iter().map(|row| row.to_vec()).collect();
        // Certified rows 0 and 2 are deliberately interleaved with unresolved
        // rows 1 and 3, whose thresholds remain distinct for alignment checks.
        let thresholds = vec![1.5_f32, -0.25, 3.5, 0.75];
        (output, spec, objectives, thresholds)
    }

    #[test]
    fn root_spec_prune_gate_off_preserves_full_request_exactly() {
        let (output, spec, objectives, thresholds) = fixture();
        let before = spec.clone();
        assert!(
            build_root_spec_prune_plan(false, false, &output, &spec, &objectives, &thresholds,)
                .is_none(),
            "gate-off must decline compression and leave the historical full request in force"
        );
        assert_eq!(
            spec, before,
            "planning must not mutate the full spec matrix"
        );

        assert!(
            build_root_spec_prune_plan(true, true, &output, &spec, &objectives, &thresholds,)
                .is_none(),
            "conjunctive root semantics must always retain the full request"
        );
    }

    #[test]
    fn root_spec_prune_active_matrix_keeps_only_unverified_rows_in_order() {
        let (output, spec, objectives, thresholds) = fixture();
        let plan =
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .expect("valid disjunctive fixture should produce a compression plan");

        assert_eq!(plan.active_indices, vec![1]);
        let active = plan
            .active_spec_matrix
            .expect("one unverified objective must produce one active row");
        assert_eq!(active.nrows(), 1);
        assert_eq!(active.ncols(), 2);
        assert_eq!(active.row(0), spec.row(1));
    }

    #[test]
    fn root_spec_prune_exact_c_compacts_noncontiguous_rows_with_one_alignment_map() {
        let (output, spec, objectives, thresholds) = noncontiguous_fixture();
        let plan =
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .expect("valid noncontiguous plan");
        let compact = build_atomic_root_c_compact_rows(&plan, &spec, &objectives, &thresholds)
            .expect("strictly aligned compact exact-C rows");

        assert_eq!(compact.source_indices, vec![1, 3]);
        assert_eq!(
            compact.objectives,
            vec![objectives[1].clone(), objectives[3].clone()]
        );
        assert_eq!(compact.thresholds, vec![-0.25, 0.75]);
        assert_eq!(compact.spec_matrix.row(0), spec.row(1));
        assert_eq!(compact.spec_matrix.row(1), spec.row(3));
        assert_eq!(
            compact
                .reference
                .lower()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![plan.pre_bounds[1].0, plan.pre_bounds[3].0]
        );
        assert_eq!(
            compact
                .reference
                .upper()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![plan.pre_bounds[1].1, plan.pre_bounds[3].1]
        );
    }

    #[test]
    fn root_spec_prune_exact_c_rejects_reordered_duplicate_or_drifted_source_maps() {
        let (output, spec, objectives, thresholds) = noncontiguous_fixture();
        let make_plan = || {
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .expect("valid plan")
        };

        let mut reordered = make_plan();
        reordered.active_indices = vec![3, 1];
        assert!(
            build_atomic_root_c_compact_rows(&reordered, &spec, &objectives, &thresholds).is_none()
        );

        let mut duplicate = make_plan();
        duplicate.active_indices = vec![1, 1];
        assert!(
            build_atomic_root_c_compact_rows(&duplicate, &spec, &objectives, &thresholds).is_none()
        );

        let mut out_of_range = make_plan();
        out_of_range.active_indices = vec![1, 4];
        assert!(
            build_atomic_root_c_compact_rows(&out_of_range, &spec, &objectives, &thresholds)
                .is_none()
        );

        let mut drifted_c = make_plan();
        drifted_c.active_spec_matrix.as_mut().expect("active C")[[0, 0]] = 1.0;
        assert!(
            build_atomic_root_c_compact_rows(&drifted_c, &spec, &objectives, &thresholds).is_none()
        );
    }

    #[test]
    fn root_spec_prune_merge_restores_the_full_bound_vector_without_reordering() {
        let (output, spec, objectives, thresholds) = fixture();
        let plan =
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .unwrap();
        let active_result = (2.25_f32, 3.5_f32);
        let merged = merge_root_spec_pruned_bounds(&plan, vec![active_result])
            .expect("one active row should merge into a three-row full vector");

        assert_eq!(merged.len(), objectives.len());
        assert_eq!(merged[0], plan.pre_bounds[0]);
        assert_eq!(merged[1], active_result);
        assert_eq!(merged[2], plan.pre_bounds[2]);
        assert!(merge_root_spec_pruned_bounds(&plan, Vec::new()).is_none());

        let looser = merge_root_spec_pruned_bounds(&plan, vec![(1.0, 5.0)]).unwrap();
        assert_eq!(
            looser[1], plan.pre_bounds[1],
            "a looser active candidate must not weaken the bootstrap row"
        );
        assert!(
            merge_root_spec_pruned_bounds(&plan, vec![(10.0, 11.0)]).is_none(),
            "a disjoint active candidate must reject the compact result and cache"
        );
        assert!(
            merge_root_spec_pruned_bounds(&plan, vec![(f32::NAN, 3.5)]).is_none(),
            "a malformed active candidate must reject the compact result and cache"
        );

        for malformed_pre in [(f32::INFINITY, f32::INFINITY), (f32::NAN, f32::NAN)] {
            let mut malformed_plan =
                build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                    .unwrap();
            malformed_plan.pre_bounds[1] = malformed_pre;
            assert!(
                merge_root_spec_pruned_bounds(
                    &malformed_plan,
                    vec![(f32::INFINITY, f32::INFINITY)]
                )
                .is_none(),
                "an unusable prebound and unusable active result must retry the full request"
            );
            let recovered =
                merge_root_spec_pruned_bounds(&malformed_plan, vec![active_result]).unwrap();
            assert_eq!(
                recovered[1], active_result,
                "a fresh finite active enclosure must replace a malformed prebound"
            );
        }
    }

    #[test]
    fn root_spec_prune_merge_preserves_certified_row_bits_across_noncontiguous_restore() {
        let (output, spec, objectives, thresholds) = noncontiguous_fixture();
        let mut plan =
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .expect("valid noncontiguous plan");
        plan.pre_bounds[0] = (f32::from_bits(0x3f80_0001), f32::from_bits(0x4040_0001));
        plan.pre_bounds[2] = (f32::from_bits(0x4080_0001), f32::from_bits(0x40a0_0001));
        let certified_bits = [
            (
                plan.pre_bounds[0].0.to_bits(),
                plan.pre_bounds[0].1.to_bits(),
            ),
            (
                plan.pre_bounds[2].0.to_bits(),
                plan.pre_bounds[2].1.to_bits(),
            ),
        ];

        let merged =
            merge_root_spec_pruned_bounds(&plan, vec![(-1.5_f32, 0.5_f32), (-3.0_f32, 1.25_f32)])
                .expect("aligned active rows restore");
        assert_eq!(
            (merged[0].0.to_bits(), merged[0].1.to_bits()),
            certified_bits[0]
        );
        assert_eq!(merged[1], (-1.5, 0.5));
        assert_eq!(
            (merged[2].0.to_bits(), merged[2].1.to_bits()),
            certified_bits[1]
        );
        assert_eq!(merged[3], (-3.0, 1.25));
    }

    #[test]
    fn root_spec_prune_committed_compact_merge_failure_never_retries_full_crown() {
        assert!(retry_full_spec_after_compact_merge_failure(false));
        assert!(!retry_full_spec_after_compact_merge_failure(true));
    }

    #[test]
    fn committed_compact_reconstruction_failure_publishes_only_bootstrap_authority() {
        let (output, spec, objectives, thresholds) = fixture();
        let plan =
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .expect("valid compact plan");
        let expected_bits = plan
            .pre_bounds
            .iter()
            .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
            .collect::<Vec<_>>();

        let publication = finalize_root_spec_prune_publication(
            &plan,
            vec![(10.0, 11.0)],
            Some(GraphAlphaState::new()),
            true,
            true,
        );

        assert!(!publication.retry_full_spec);
        assert!(!publication.bounds_reconstruction_succeeded);
        assert!(publication.alpha.is_none());
        assert_eq!(
            publication
                .bounds
                .expect("bootstrap bounds remain authoritative")
                .iter()
                .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
                .collect::<Vec<_>>(),
            expected_bits
        );
    }

    #[test]
    fn root_spec_prune_selected_alpha_stays_pending_until_compact_reconstruction() {
        assert!(
            publish_pending_atomic_root_alpha(Some(GraphAlphaState::new()), true, false,).is_none()
        );
        assert!(
            publish_pending_atomic_root_alpha(Some(GraphAlphaState::new()), true, true,).is_some()
        );
        assert!(
            publish_pending_atomic_root_alpha(Some(GraphAlphaState::new()), false, false,)
                .is_some()
        );
    }

    #[test]
    fn root_spec_prune_binding_rows_map_back_to_noncontiguous_source_rows() {
        let source_indices = [1usize, 3];
        assert_eq!(map_compact_atomic_root_c_row(&source_indices, 0), Some(1));
        assert_eq!(map_compact_atomic_root_c_row(&source_indices, 1), Some(3));
        assert_eq!(map_compact_atomic_root_c_row(&source_indices, 2), None);
        assert_eq!(
            map_compact_atomic_root_c_rows(&source_indices, &[1, 0, 1]),
            Some(vec![3, 1, 3])
        );
        assert_eq!(
            map_compact_atomic_root_c_rows(&source_indices, &[0, 2]),
            None
        );
        assert!(compact_atomic_root_c_reconstruction_succeeded(
            true, true, true,
        ));
        assert!(
            !compact_atomic_root_c_reconstruction_succeeded(true, true, false),
            "a committed bounds vector cannot publish when binding provenance is out of range"
        );
        assert!(
            !compact_atomic_root_c_reconstruction_succeeded(true, false, true),
            "a valid binding map cannot replace failed bounds reconstruction"
        );
    }

    #[test]
    fn root_spec_prune_all_pruned_skips_the_active_matrix() {
        let (output, spec, objectives, _) = fixture();
        let thresholds = vec![0.0_f32, 1.0, -4.0];
        let plan =
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .expect("all-pruned fixture should still produce its certified full result");

        assert!(plan.active_indices.is_empty());
        assert!(plan.active_spec_matrix.is_none());
        assert_eq!(plan.pre_bounds.len(), objectives.len());
        for ((lower, _upper), threshold) in plan.pre_bounds.iter().zip(thresholds) {
            assert!(*lower > threshold);
        }
    }

    #[test]
    fn root_spec_prune_malformed_prebound_endpoints_never_certify() {
        assert!(root_prebound_certifies(1.0, 2.0, 0.0));
        assert!(!root_prebound_certifies(1.0, f32::NAN, 0.0));
        assert!(!root_prebound_certifies(1.0, f32::INFINITY, 0.0));
        assert!(!root_prebound_certifies(2.0, 1.0, 0.0));
        assert!(!root_prebound_certifies(f32::INFINITY, f32::INFINITY, 0.0));
    }

    #[test]
    fn root_spec_prune_cache_rows_expand_to_their_original_objective_slots() {
        let mut compact = CachedLinearBounds::default();
        compact
            .lower_a
            .insert("relu".to_string(), arr2(&[[10.0_f32, 11.0], [30.0, 31.0]]));
        compact
            .upper_a
            .insert("relu".to_string(), arr2(&[[12.0_f32, 13.0], [32.0, 33.0]]));
        compact
            .lower_b
            .insert("relu".to_string(), arr1(&[100.0_f32, 300.0]));
        compact
            .upper_b
            .insert("relu".to_string(), arr1(&[101.0_f32, 301.0]));

        let expanded = expand_root_spec_cache(&compact, &[1, 3], 4)
            .expect("two compact rows should map into four full objective slots");
        assert_eq!(expanded.len(), 4);
        assert!(expanded[0].is_none());
        assert!(expanded[2].is_none());
        assert_eq!(
            expanded[1]
                .as_ref()
                .and_then(|cache| cache.lower_a.get("relu"))
                .map(|a| a[[0, 0]]),
            Some(10.0)
        );
        assert_eq!(
            expanded[3]
                .as_ref()
                .and_then(|cache| cache.lower_a.get("relu"))
                .map(|a| a[[0, 0]]),
            Some(30.0)
        );
        assert!(expand_root_spec_cache(&compact, &[1, 1], 4).is_none());
        assert!(expand_root_spec_cache(&compact, &[1, 4], 4).is_none());
        assert!(
            expand_root_spec_cache(&compact, &[1], 4).is_none(),
            "a sparse one-row mapping must reject an unexpectedly two-row cache"
        );
    }
}

/// Choose the root objective authority as one atomic value. Keeping this seam
/// pure makes the fail-open contract explicit: when Stage B is disabled or
/// rejects, the exact Stage-A bounds, cache, and row map are moved through
/// unchanged. No per-row mutation occurs at this layer.
fn select_post_c_survivor_or_stage_a(
    stage_a_bounds: Vec<(f32, f32)>,
    stage_a_cache: Option<CachedLinearBounds>,
    stage_a_active_indices: Vec<usize>,
    post_c_survivor: Option<PostCSurvivorAccepted>,
) -> (Vec<(f32, f32)>, Option<CachedLinearBounds>, Vec<usize>) {
    match post_c_survivor {
        Some(post_c) => (
            post_c.merged_bounds,
            Some(post_c.compact_cache),
            post_c.active_indices,
        ),
        None => (stage_a_bounds, stage_a_cache, stage_a_active_indices),
    }
}

#[cfg(test)]
mod post_c_root_publication_tests {
    use super::select_post_c_survivor_or_stage_a;
    use crate::batched_domain::CachedLinearBounds;
    use ndarray::{arr1, arr2};

    #[test]
    fn disabled_or_refused_stage_b_preserves_stage_a_bounds_cache_and_map_bit_exactly() {
        let stage_a_bounds = vec![
            (f32::from_bits(0x3f80_0001), f32::from_bits(0x4000_0001)),
            (f32::from_bits(0xbf00_0001), f32::from_bits(0x4040_0001)),
        ];
        let expected_bits: Vec<_> = stage_a_bounds
            .iter()
            .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
            .collect();
        let mut stage_a_cache = CachedLinearBounds::default();
        stage_a_cache
            .lower_a
            .insert("relu".to_string(), arr2(&[[1.25_f32, -2.5], [3.75, -4.0]]));
        stage_a_cache
            .upper_a
            .insert("relu".to_string(), arr2(&[[5.25_f32, -6.5], [7.75, -8.0]]));
        stage_a_cache
            .lower_b
            .insert("relu".to_string(), arr1(&[0.125_f32, -0.25]));
        stage_a_cache
            .upper_b
            .insert("relu".to_string(), arr1(&[0.5_f32, -1.0]));

        let (bounds, cache, active_indices) = select_post_c_survivor_or_stage_a(
            stage_a_bounds,
            Some(stage_a_cache),
            vec![0, 1],
            None,
        );
        assert_eq!(
            bounds
                .iter()
                .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
                .collect::<Vec<_>>(),
            expected_bits
        );
        assert_eq!(active_indices, vec![0, 1]);
        let cache = cache.expect("Stage-A cache must survive a Stage-B refusal");
        assert_eq!(cache.lower_a["relu"], arr2(&[[1.25, -2.5], [3.75, -4.0]]));
        assert_eq!(cache.upper_a["relu"], arr2(&[[5.25, -6.5], [7.75, -8.0]]));
        assert_eq!(cache.lower_b["relu"], arr1(&[0.125, -0.25]));
        assert_eq!(cache.upper_b["relu"], arr1(&[0.5, -1.0]));
    }
}

#[cfg(test)]
std::thread_local! {
    static ROOT_OBJECTIVE_SPEC_BUILD_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_root_objective_spec_build_count_for_test() {
    ROOT_OBJECTIVE_SPEC_BUILD_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(super) fn root_objective_spec_build_count_for_test() -> usize {
    ROOT_OBJECTIVE_SPEC_BUILD_COUNT.with(std::cell::Cell::get)
}

pub(super) fn compute_root_objective_bounds(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    conjunctive: bool,
    engine: Option<&dyn GemmEngine>,
    bootstrap: &GraphBabBootstrap,
    global_deadline: Option<std::time::Instant>,
    root_intermediate_bounds_changed: bool,
    root_dense_head_stage_selected: bool,
) -> Result<RootObjectiveEvaluation> {
    // The outer verifier authority is terminal. Check it before even scanning
    // objective rows or allocating the dense C matrix.
    if global_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
        return Err(ny_core::NyError::DeadlineExceeded(
            "multi-objective root authority expired before spec construction".to_string(),
        ));
    }
    #[cfg(test)]
    ROOT_OBJECTIVE_SPEC_BUILD_COUNT.with(|count| count.set(count.get() + 1));
    let spec_matrix = match global_deadline {
        Some(_) if objectives.is_empty() => None,
        Some(deadline) => {
            match crate::beta_crown::engine::graph::objectives::build_spec_matrix_for_authority(
                objectives,
                Some(deadline),
            ) {
                Ok(matrix) => Some(matrix),
                Err(ny_core::NyError::InvalidSpec(_)) => None,
                Err(error) => return Err(error),
            }
        }
        None => build_spec_matrix(objectives),
    };
    let mut root_spec_cache = None;
    let mut root_spec_cache_active_indices = Vec::new();
    let mut root_alpha_override = None;
    let root_spec_prune_configured = root_spec_prune_enabled();
    crate::execution_telemetry::record_root_spec_prune_route(root_spec_prune_configured);
    // Keep gate-off and conjunctive execution on the exact historic path: do
    // not even resolve/inspect the bootstrap output unless compression is armed.
    let root_spec_prune_plan = if root_spec_prune_configured && !conjunctive {
        spec_matrix.as_ref().and_then(|full_spec_matrix| {
            let output = bootstrap_output_bounds(graph, bootstrap)?;
            build_root_spec_prune_plan(
                true,
                false,
                output,
                full_spec_matrix,
                objectives,
                thresholds,
            )
        })
    } else {
        None
    };
    if let Some(plan) = root_spec_prune_plan.as_ref() {
        crate::execution_telemetry::record_root_spec_prune_plan(
            objectives.len(),
            plan.active_indices.len(),
            objectives.len() - plan.active_indices.len(),
        );
    }

    // Root-pass grace slice (#w4-root-gpu): the alpha warmup runs to the FULL
    // initial deadline (measured on cifar100: `bootstrap.alpha_config.deadline`
    // is already EXPIRED by the time this root pass runs), so the C-matrix root
    // pass — the root-verification lever, <1s on the sound GPU resnet backward —
    // refused to start and the root objective bounds degraded to the per-logit
    // IBP projection every time. Grant the root spec pass a small grace slice
    // when its deadline has been consumed, capped by the caller's global
    // wall-clock deadline. Sound: deadlines only schedule work; the bounds
    // computed under the grace slice are the same certified machinery.
    //
    // #w4-root-alpha-opt: on conv DAGs the root pass additionally runs the
    // forward-map alpha OPTIMIZER (cheap surrogate sweeps) followed by ONE
    // alpha-fed rebuild of the forward-linear map — a full O(L) certified
    // pass (~22s on cifar100 release), the ROOT-relaxation lever. It needs no
    // warmup alphas (it starts from the adaptive slopes and optimizes the
    // forward objective directly). That work gets a larger grace: a slice of
    // the remaining global budget, capped at 30s. If it does not fit, the
    // optimizer fail-closes and the fixed-slope root candidates stand (never
    // uncapped work past the timeout — #4260 regression contract).
    const ROOT_SPEC_GRACE: std::time::Duration = std::time::Duration::from_secs(3);
    const ROOT_SPEC_ALPHA_GRACE_CAP: std::time::Duration = std::time::Duration::from_secs(40);
    let now = std::time::Instant::now();
    let alpha_rebuild_pending =
        graph.has_conv_layers() && graph.forward_linear_spec_alpha_enabled();
    let grace_slice = if alpha_rebuild_pending {
        // 9/10 of the remaining global budget, cap 40s (measured at 95s with
        // the warmup cap above: remaining ≈ 35-43s at root; the optimizer +
        // rebuild need ~1.15x measured fixed cost + sweeps ≈ 30s+; the old
        // 0.8/30s grace left 22s and the optimizer could never fire). The
        // spec core skips the optimizer+rebuild entirely when the measured
        // cost does not fit this slice, so unused grace returns to BaB.
        let global_remaining = global_deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or(ROOT_SPEC_ALPHA_GRACE_CAP);
        global_remaining
            .mul_f32(0.9)
            .min(ROOT_SPEC_ALPHA_GRACE_CAP)
            .max(ROOT_SPEC_GRACE)
    } else {
        ROOT_SPEC_GRACE
    };
    // Live warmups keep their historical deadline unless the alpha rebuild
    // needs the bounded grace. Expired local phase checkpoints rebase only
    // while the GLOBAL wall-clock still has room; exhausted authority never
    // mints fresh time (#4260 regression contract).
    let root_deadline = resolve_root_objective_deadline(
        bootstrap.alpha_config.deadline,
        now,
        grace_slice,
        global_deadline,
        alpha_rebuild_pending,
    );

    // NY_SLACK_PROBE: clear the per-row f32-slack accumulator so the report below
    // reflects only this root backward (dark; no-op when the gate is off).
    if crate::bounds::slack_probe_enabled() {
        let _ = crate::bounds::slack_probe_take();
    }
    let (initial_output, initial_obj_bounds) = if let Some(all_pruned) = root_spec_prune_plan
        .as_ref()
        .filter(|plan| plan.active_indices.is_empty())
    {
        info!(
            "Multi-objective: root spec pre-prune certified all {} objectives; skipping spec-guided CROWN",
            objectives.len(),
        );
        crate::execution_telemetry::record_root_spec_prune_applied(
            objectives.len(),
            0,
            objectives.len(),
            true,
            false,
        );
        (
            all_pruned.bootstrap_output.clone(),
            all_pruned.pre_bounds.clone(),
        )
    } else if let Some(ref full_spec_mat) = spec_matrix {
        let mut applied_prune = root_spec_prune_plan.as_ref();
        let selected_spec_mat = applied_prune
            .and_then(|plan| plan.active_spec_matrix.as_ref())
            .unwrap_or(full_spec_mat);

        if let Some(plan) = applied_prune {
            info!(
                "Multi-objective: root spec pre-prune kept {} of {} objectives",
                plan.active_indices.len(),
                objectives.len(),
            );
        }

        // Tightening an intermediate box invalidates the QUALITY assumptions of
        // the warmup alpha (not its soundness). On prop1761 the inherited alpha
        // loses 0.53 margin after the dense-head box changes. Preserve the
        // historical request first, then—only with deadline left—sound-intersect
        // an adaptive full DAG backward. Unchanged boxes execute the original
        // request below exactly once.
        let run_spec_request = |spec_mat: &ndarray::Array2<f32>| {
            if root_intermediate_bounds_changed {
                let historical = SpecCrownRequest::new(graph, input, spec_mat, engine)
                    .node_bounds(&bootstrap.initial_node_bounds)
                    .alpha_state_opt(bootstrap.root_alpha_state.as_ref())
                    .deadline_opt(root_deadline)
                    .truncate_after_opt(verifier.config.crown_backward_layers)
                    .capture_cache()
                    .run_with_cache();
                // #root-adaptive-spec (NY_ROOT_ADAPTIVE_SPEC=1, default OFF):
                // the SECOND (adaptive, fresh-slope) full coefficient backward
                // is an opt-in quality experiment. The historical candidate
                // already consumes the tightened head node bounds and, on the
                // deep image surface, finishes via the sound GPU root route.
                // SOUND: the adaptive pass only INTERSECTS extra tightening via
                // best_of_sound_root_spec_bounds; dropping it can only LOOSEN
                // the root box, never emit a wrong verdict. Defaulting it off
                // preserves the remaining slice for authoritative BaB instead
                // of risking repeated dense CPU fallback materialization.
                if root_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
                    || !root_adaptive_spec_enabled()
                {
                    historical
                } else {
                    let adaptive = run_adaptive_root_spec_candidate(
                        graph,
                        input,
                        spec_mat,
                        engine,
                        &bootstrap.initial_node_bounds,
                        None,
                        root_deadline,
                        verifier.config.crown_backward_layers,
                    );
                    match (historical, adaptive) {
                        (
                            Ok((historical_bounds, _historical_cache)),
                            Ok((adaptive_bounds, cache)),
                        ) => {
                            // The cache is itself a certified linear enclosure and
                            // child consumers use it only as a same-row backward
                            // seed. It stays sound even when the independent
                            // historical candidate tightens the displayed root box.
                            Ok((
                                best_of_sound_root_spec_bounds(adaptive_bounds, &historical_bounds),
                                cache,
                            ))
                        }
                        (Ok(result), Err(error)) => {
                            debug!(
                                %error,
                                "Post-tightening adaptive root backward unavailable; retaining historical root candidate"
                            );
                            Ok(result)
                        }
                        (Err(error), Ok(result)) => {
                            debug!(
                                %error,
                                "Post-tightening historical root candidate unavailable; retaining adaptive backward"
                            );
                            Ok(result)
                        }
                        (Err(historical_error), Err(adaptive_error)) => {
                            debug!(
                                %adaptive_error,
                                "Post-tightening adaptive root backward also unavailable"
                            );
                            Err(historical_error)
                        }
                    }
                }
            } else {
                SpecCrownRequest::new(graph, input, spec_mat, engine)
                    .node_bounds(&bootstrap.initial_node_bounds)
                    .alpha_state_opt(bootstrap.root_alpha_state.as_ref())
                    .deadline_opt(root_deadline)
                    .truncate_after_opt(verifier.config.crown_backward_layers)
                    .capture_cache()
                    .run_with_cache()
            }
        };

        // #root-alpha-cuda-rows: an exact-dark Stage-A sibling for the root C
        // matrix. Typed multi-iteration requests may compose with root pruning
        // only through one validated carrier that binds every active objective,
        // threshold, C row, reference interval, and source-row identity. Legacy
        // rows and legacy one-step routes retain their compression refusal.
        // The effective deadline is the intersection of the root grace and the
        // authoritative caller/global boundary derived by `evaluate_root`.
        //
        // Before backend commitment every refusal preserves the historical
        // request byte path. After commitment, either the complete selected-row
        // CUDA vector reconstructed with the omitted rows' certified bootstrap
        // bounds, or that complete bootstrap vector itself, is authoritative:
        // no ordinary GPU/CPU fallback request may run.
        let typed_margin_iterations = verifier.config.atomic_root_c_margin_iterations;
        let legacy_rows_enabled = root_alpha_cuda_rows_enabled();
        // Preserve the historical subordinate lookup: with typed iterations
        // dark and the rows parent off, the legacy margin gate is not read.
        let legacy_margin_enabled = legacy_rows_enabled && root_alpha_cuda_margin_step_enabled();
        let atomic_root_c_route = select_atomic_root_c_route(
            typed_margin_iterations,
            legacy_rows_enabled,
            legacy_margin_enabled,
        );
        let atomic_root_c_compact_rows = if typed_margin_iterations > 0 {
            applied_prune.and_then(|plan| {
                build_atomic_root_c_compact_rows(plan, full_spec_mat, objectives, thresholds)
            })
        } else {
            None
        };
        let atomic_evaluated_rows = atomic_root_c_compact_rows
            .as_ref()
            .map_or(full_spec_mat.nrows(), |compact| compact.spec_matrix.nrows());
        let exact_c_compressed_selected =
            typed_margin_iterations > 0 && atomic_root_c_compact_rows.is_some();
        let exact_c_precertified_rows = objectives.len() - atomic_evaluated_rows;
        let atomic_root_c =
            atomic_root_c_route.map(|route| {
                let compact_margin_route = atomic_root_c_route_accepts_compact_rows(
                    route,
                    atomic_root_c_compact_rows.is_some(),
                );
                if applied_prune.is_some() && !compact_margin_route {
                    let refusal = AtomicCudaRowsRefusal::SpecCompressionActive;
                    return match route {
                        AtomicRootCRoute::Rows => AtomicRootCStageAOutcome::Rows(
                            AtomicCudaRowsOutcome::RefusedBeforeCommit { refusal },
                        ),
                        AtomicRootCRoute::Margin { .. } => AtomicRootCStageAOutcome::MarginStep(
                            AtomicCudaMarginStepOutcome::RefusedBeforeCommit { refusal },
                        ),
                    };
                }

                let authority_deadline = match (root_deadline, global_deadline) {
                    (Some(root), Some(global)) => Some(root.min(global)),
                    (Some(root), None) => Some(root),
                    (None, global) => global,
                };
                match route {
                    AtomicRootCRoute::Rows => {
                        AtomicRootCStageAOutcome::Rows(run_atomic_root_c_stage_a(
                            graph,
                            input,
                            objectives,
                            full_spec_mat,
                            bootstrap,
                            authority_deadline,
                        ))
                    }
                    AtomicRootCRoute::Margin { multi_iterations } => {
                        let (route_objectives, route_thresholds, route_spec_matrix, reference) =
                            match atomic_root_c_compact_rows.as_ref() {
                                Some(compact) => (
                                    compact.objectives.as_slice(),
                                    compact.thresholds.as_slice(),
                                    &compact.spec_matrix,
                                    Some(&compact.reference),
                                ),
                                None => (objectives, thresholds, full_spec_mat, None),
                            };
                        AtomicRootCStageAOutcome::MarginStep(run_atomic_root_c_margin_step(
                            graph,
                            input,
                            route_objectives,
                            route_thresholds,
                            route_spec_matrix,
                            reference,
                            bootstrap,
                            conjunctive,
                            verifier.config.verify_upper_bound,
                            multi_iterations,
                            authority_deadline,
                        ))
                    }
                }
            });
        if typed_margin_iterations > 0 {
            crate::execution_telemetry::record_exact_c_selected(
                typed_margin_iterations,
                objectives.len(),
                atomic_evaluated_rows,
                exact_c_precertified_rows,
            );
            if let Some(outcome) = atomic_root_c.as_ref() {
                observe_typed_exact_c_outcome(outcome);
            }
        }
        let source_binding_row = |compact_row: usize| match atomic_root_c_compact_rows.as_ref() {
            Some(compact) => map_compact_atomic_root_c_row(&compact.source_indices, compact_row),
            None => Some(compact_row),
        };
        let source_binding_rows = |compact_rows: &[usize]| match atomic_root_c_compact_rows.as_ref()
        {
            Some(compact) => map_compact_atomic_root_c_rows(&compact.source_indices, compact_rows),
            None => Some(compact_rows.to_vec()),
        };
        // A selected alpha remains private until any compact result has been
        // restored into a complete source-ordered bound vector.
        let mut pending_root_alpha_override = None;
        let mut atomic_binding_map_valid = true;
        let (mut spec_result, atomic_stage_a_committed) = match atomic_root_c {
            None => (run_spec_request(selected_spec_mat), false),
            Some(AtomicRootCStageAOutcome::Rows(AtomicCudaRowsOutcome::RefusedBeforeCommit {
                refusal: AtomicCudaRowsRefusal::DeadlineExceeded,
            }))
            | Some(AtomicRootCStageAOutcome::MarginStep(
                AtomicCudaMarginStepOutcome::RefusedBeforeCommit {
                    refusal: AtomicCudaRowsRefusal::DeadlineExceeded,
                },
            )) => {
                if exact_c_compressed_selected {
                    crate::execution_telemetry::record_exact_c_compressed_selection_rolled_back(
                        objectives.len(),
                        atomic_evaluated_rows,
                        exact_c_precertified_rows,
                    );
                }
                return Err(ny_core::NyError::DeadlineExceeded(
                    "atomic CUDA root-C deadline exceeded before commitment".to_string(),
                ));
            }
            Some(AtomicRootCStageAOutcome::Rows(AtomicCudaRowsOutcome::RefusedBeforeCommit {
                refusal,
            }))
            | Some(AtomicRootCStageAOutcome::MarginStep(
                AtomicCudaMarginStepOutcome::RefusedBeforeCommit { refusal },
            )) => {
                info!(
                    status = "refused-before-commit",
                    stage = "root-c",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    reason = refusal.telemetry_reason(),
                    ?refusal,
                    legacy_fallback = true,
                    "Atomic CUDA rows Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-rows stage=root-c status=refused-before-commit rows={} \
                         reason={} consumer=root-objective-bounds legacy-fallback=true",
                        atomic_evaluated_rows,
                        refusal.telemetry_reason(),
                    ));
                }
                (run_spec_request(selected_spec_mat), false)
            }
            Some(AtomicRootCStageAOutcome::Rows(AtomicCudaRowsOutcome::Committed(
                AtomicCudaRowsCommit::CudaIntersection(bounds),
            ))) => {
                info!(
                    status = "accepted",
                    stage = "root-c",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    cache_published = false,
                    legacy_fallback = false,
                    "Atomic CUDA rows Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-rows stage=root-c status=accepted rows={} \
                         consumer=root-objective-bounds cache=false legacy-fallback=false",
                        atomic_evaluated_rows,
                    ));
                }
                (Ok((*bounds, None)), true)
            }
            Some(AtomicRootCStageAOutcome::Rows(AtomicCudaRowsOutcome::Committed(
                AtomicCudaRowsCommit::DeadlineExceeded,
            ))) => {
                return Err(ny_core::NyError::DeadlineExceeded(
                    "atomic CUDA root-C publication deadline exceeded".to_string(),
                ));
            }
            Some(AtomicRootCStageAOutcome::Rows(AtomicCudaRowsOutcome::Committed(
                AtomicCudaRowsCommit::ReferenceRetained {
                    bounds,
                    refusal: AtomicCudaRowsRefusal::DeadlineExceeded,
                },
            ))) => {
                drop(bounds);
                return Err(ny_core::NyError::DeadlineExceeded(
                    "atomic CUDA root-C reference deadline exceeded".to_string(),
                ));
            }
            Some(AtomicRootCStageAOutcome::Rows(AtomicCudaRowsOutcome::Committed(
                AtomicCudaRowsCommit::ReferenceRetained { bounds, refusal },
            ))) => {
                info!(
                    status = "committed-reference",
                    stage = "root-c",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    reason = refusal.telemetry_reason(),
                    ?refusal,
                    cache_published = false,
                    legacy_fallback = false,
                    "Atomic CUDA rows Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-rows stage=root-c status=committed-reference rows={} \
                         reason={} consumer=root-objective-bounds cache=false legacy-fallback=false",
                        atomic_evaluated_rows,
                        refusal.telemetry_reason(),
                    ));
                }
                (Ok((*bounds, None)), true)
            }
            Some(AtomicRootCStageAOutcome::MarginStep(AtomicCudaMarginStepOutcome::Committed(
                AtomicCudaMarginStepCommit::DeadlineExceeded,
            ))) => {
                info!(
                    status = "deadline-after-cleanup",
                    stage = "root-c-margin-step",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    cache_published = false,
                    alpha_override = false,
                    legacy_fallback = false,
                    "Atomic CUDA margin-alpha Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-margin-step status=deadline-after-cleanup rows={} \
                         cache=false alpha-override=false legacy-fallback=false",
                        atomic_evaluated_rows,
                    ));
                }
                return Err(ny_core::NyError::DeadlineExceeded(
                    "atomic CUDA margin-alpha publication deadline exceeded after cleanup"
                        .to_string(),
                ));
            }
            Some(AtomicRootCStageAOutcome::MarginStep(AtomicCudaMarginStepOutcome::Committed(
                AtomicCudaMarginStepCommit::Alpha0Retained { bounds, refusal },
            ))) => {
                info!(
                    status = "alpha0-retained",
                    stage = "root-c-margin-step",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    reason = refusal.telemetry_reason(),
                    ?refusal,
                    cache_published = false,
                    alpha_override = false,
                    legacy_fallback = false,
                    "Atomic CUDA margin-alpha Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-margin-step status=alpha0-retained rows={} reason={} \
                         cache=false alpha-override=false legacy-fallback=false",
                        atomic_evaluated_rows,
                        refusal.telemetry_reason(),
                    ));
                }
                (Ok((*bounds, None)), true)
            }
            Some(AtomicRootCStageAOutcome::MarginStep(AtomicCudaMarginStepOutcome::Committed(
                AtomicCudaMarginStepCommit::Alpha0Selected {
                    bounds,
                    binding_row,
                    alpha0_score,
                    alpha1_score,
                },
            ))) => {
                let compact_binding_row = binding_row;
                let source_binding_row = source_binding_row(compact_binding_row);
                atomic_binding_map_valid &= source_binding_row.is_some();
                info!(
                    status = "alpha0-selected",
                    stage = "root-c-margin-step",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    compact_binding_row,
                    source_binding_row = ?source_binding_row,
                    alpha0_score,
                    alpha1_score,
                    cache_published = false,
                    alpha_override = false,
                    legacy_fallback = false,
                    "Atomic CUDA margin-alpha Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-margin-step status=alpha0-selected rows={} \
                         compact-binding-row={} source-binding-row={:?} \
                         alpha0-score={alpha0_score:.6} alpha1-score={alpha1_score:.6} \
                         cache=false alpha-override=false legacy-fallback=false",
                        atomic_evaluated_rows, compact_binding_row, source_binding_row,
                    ));
                }
                (Ok((*bounds, None)), true)
            }
            Some(AtomicRootCStageAOutcome::MarginStep(AtomicCudaMarginStepOutcome::Committed(
                AtomicCudaMarginStepCommit::Alpha1Selected {
                    bounds,
                    alpha_state,
                    binding_row,
                    alpha0_score,
                    alpha1_score,
                },
            ))) => {
                let compact_binding_row = binding_row;
                let source_binding_row = source_binding_row(compact_binding_row);
                atomic_binding_map_valid &= source_binding_row.is_some();
                info!(
                    status = "alpha1-selected",
                    stage = "root-c-margin-step",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    compact_binding_row,
                    source_binding_row = ?source_binding_row,
                    alpha0_score,
                    alpha1_score,
                    cache_published = false,
                    alpha_override = false,
                    alpha_pending = true,
                    legacy_fallback = false,
                    "Atomic CUDA margin-alpha Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-margin-step status=alpha1-selected rows={} \
                         compact-binding-row={} source-binding-row={:?} \
                         alpha0-score={alpha0_score:.6} alpha1-score={alpha1_score:.6} \
                         cache=false alpha-override=false alpha-pending=true legacy-fallback=false",
                        atomic_evaluated_rows, compact_binding_row, source_binding_row,
                    ));
                }
                pending_root_alpha_override = Some(*alpha_state);
                (Ok((*bounds, None)), true)
            }
            Some(AtomicRootCStageAOutcome::MarginStep(AtomicCudaMarginStepOutcome::Committed(
                AtomicCudaMarginStepCommit::TopKAlpha0Selected {
                    bounds,
                    row_indices,
                    alpha0_score,
                    alpha1_score,
                },
            ))) => {
                let source_row_indices = source_binding_rows(row_indices.as_ref());
                atomic_binding_map_valid &= source_row_indices.is_some();
                info!(
                    status = "topk-alpha0-selected",
                    stage = "root-c-margin-step",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    topk = row_indices.len(),
                    compact_topk_rows = ?row_indices,
                    source_topk_rows = ?source_row_indices,
                    alpha0_score,
                    alpha1_score,
                    cache_published = false,
                    alpha_override = false,
                    legacy_fallback = false,
                    "Atomic CUDA top-K margin-alpha Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-margin-step status=topk-alpha0-selected rows={} \
                         policy=topk-sum topk={} compact-topk-rows={row_indices:?} \
                         source-topk-rows={source_row_indices:?} \
                         alpha0-score={alpha0_score:.6} alpha1-score={alpha1_score:.6} \
                         cache=false alpha-override=false legacy-fallback=false",
                        atomic_evaluated_rows,
                        row_indices.len(),
                    ));
                }
                (Ok((*bounds, None)), true)
            }
            Some(AtomicRootCStageAOutcome::MarginStep(AtomicCudaMarginStepOutcome::Committed(
                AtomicCudaMarginStepCommit::TopKAlpha1Selected {
                    bounds,
                    alpha_state,
                    row_indices,
                    alpha0_score,
                    alpha1_score,
                },
            ))) => {
                let source_row_indices = source_binding_rows(row_indices.as_ref());
                atomic_binding_map_valid &= source_row_indices.is_some();
                info!(
                    status = "topk-alpha1-selected",
                    stage = "root-c-margin-step",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    topk = row_indices.len(),
                    compact_topk_rows = ?row_indices,
                    source_topk_rows = ?source_row_indices,
                    alpha0_score,
                    alpha1_score,
                    cache_published = false,
                    alpha_override = false,
                    alpha_pending = true,
                    legacy_fallback = false,
                    "Atomic CUDA top-K margin-alpha Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-margin-step status=topk-alpha1-selected rows={} \
                         policy=topk-sum topk={} compact-topk-rows={row_indices:?} \
                         source-topk-rows={source_row_indices:?} \
                         alpha0-score={alpha0_score:.6} alpha1-score={alpha1_score:.6} \
                         cache=false alpha-override=false alpha-pending=true legacy-fallback=false",
                        atomic_evaluated_rows,
                        row_indices.len(),
                    ));
                }
                pending_root_alpha_override = Some(*alpha_state);
                (Ok((*bounds, None)), true)
            }
            Some(AtomicRootCStageAOutcome::MarginStep(AtomicCudaMarginStepOutcome::Committed(
                AtomicCudaMarginStepCommit::MultiAlpha0Selected {
                    bounds,
                    binding_rows,
                    attempted_iterations,
                    multiplicative_weights_requested,
                    multiplicative_weights_plan_dispatched,
                    multiplicative_weights_effective,
                    completed_proposals,
                    adaptive_plan_dispatches,
                    gradient_plan_num_specs,
                    gradient_row_count,
                    initial_score,
                    final_score,
                    stop_refusal,
                },
            ))) => {
                let stop_reason = stop_refusal
                    .map(|refusal| refusal.telemetry_reason())
                    .unwrap_or("iteration_limit");
                let source_binding_rows = source_binding_rows(binding_rows.as_ref());
                atomic_binding_map_valid &= source_binding_rows.is_some();
                info!(
                    status = "multi-alpha0-selected",
                    stage = "root-c-margin-step",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    attempted_iterations,
                    accepted_iterations = 0,
                    completed_proposals,
                    adaptive_plan_dispatches,
                    multiplicative_weights_requested,
                    multiplicative_weights_plan_dispatched,
                    multiplicative_weights_effective,
                    gradient_plan_num_specs = ?gradient_plan_num_specs,
                    gradient_row_count,
                    compact_binding_rows = ?binding_rows,
                    source_binding_rows = ?source_binding_rows,
                    initial_score,
                    final_score,
                    stop_reason,
                    cache_published = false,
                    alpha_override = false,
                    legacy_fallback = false,
                    "Atomic CUDA bounded multi-iteration margin-alpha Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-margin-step status=multi-alpha0-selected rows={} \
                         attempted-iterations={} completed-proposals={} \
                         adaptive-plan-dispatches={} \
                         accepted-iterations=0 mw-requested={} \
                         mw-plan-dispatched={} mw-effective={} \
                         gradient-plan-num-specs={gradient_plan_num_specs:?} \
                         gradient-row-count={} \
                         compact-binding-rows={binding_rows:?} \
                         source-binding-rows={source_binding_rows:?} \
                         initial-score={initial_score:.6} \
                         final-score={final_score:.6} stop-reason={stop_reason} \
                         cache=false alpha-override=false legacy-fallback=false",
                        atomic_evaluated_rows,
                        attempted_iterations,
                        completed_proposals,
                        adaptive_plan_dispatches,
                        multiplicative_weights_requested,
                        multiplicative_weights_plan_dispatched,
                        multiplicative_weights_effective,
                        gradient_row_count,
                    ));
                }
                (Ok((*bounds, None)), true)
            }
            Some(AtomicRootCStageAOutcome::MarginStep(AtomicCudaMarginStepOutcome::Committed(
                AtomicCudaMarginStepCommit::MultiAlphaSelected {
                    bounds,
                    alpha_state,
                    binding_rows,
                    attempted_iterations,
                    accepted_iterations,
                    multiplicative_weights_requested,
                    multiplicative_weights_plan_dispatched,
                    multiplicative_weights_effective,
                    completed_proposals,
                    adaptive_plan_dispatches,
                    gradient_plan_num_specs,
                    gradient_row_count,
                    initial_score,
                    final_score,
                    stop_refusal,
                },
            ))) => {
                let stop_reason = stop_refusal
                    .map(|refusal| refusal.telemetry_reason())
                    .unwrap_or("iteration_limit");
                let source_binding_rows = source_binding_rows(binding_rows.as_ref());
                atomic_binding_map_valid &= source_binding_rows.is_some();
                info!(
                    status = "multi-alpha-selected",
                    stage = "root-c-margin-step",
                    consumer = "root-objective-bounds",
                    rows = atomic_evaluated_rows,
                    attempted_iterations,
                    accepted_iterations,
                    completed_proposals,
                    adaptive_plan_dispatches,
                    multiplicative_weights_requested,
                    multiplicative_weights_plan_dispatched,
                    multiplicative_weights_effective,
                    gradient_plan_num_specs = ?gradient_plan_num_specs,
                    gradient_row_count,
                    compact_binding_rows = ?binding_rows,
                    source_binding_rows = ?source_binding_rows,
                    initial_score,
                    final_score,
                    stop_reason,
                    cache_published = false,
                    alpha_override = false,
                    alpha_pending = true,
                    legacy_fallback = false,
                    "Atomic CUDA bounded multi-iteration margin-alpha Stage-A"
                );
                if crate::phase_telemetry::phase_telemetry_enabled() {
                    crate::phase_telemetry::phase_marker(&format!(
                        "root-cuda-margin-step status=multi-alpha-selected rows={} \
                         attempted-iterations={} completed-proposals={} \
                         adaptive-plan-dispatches={} \
                         accepted-iterations={} mw-requested={} \
                         mw-plan-dispatched={} mw-effective={} \
                         gradient-plan-num-specs={gradient_plan_num_specs:?} \
                         gradient-row-count={} \
                         compact-binding-rows={binding_rows:?} \
                         source-binding-rows={source_binding_rows:?} \
                         initial-score={initial_score:.6} \
                         final-score={final_score:.6} stop-reason={stop_reason} \
                         cache=false alpha-override=false alpha-pending=true legacy-fallback=false",
                        atomic_evaluated_rows,
                        attempted_iterations,
                        completed_proposals,
                        adaptive_plan_dispatches,
                        accepted_iterations,
                        multiplicative_weights_requested,
                        multiplicative_weights_plan_dispatched,
                        multiplicative_weights_effective,
                        gradient_row_count,
                    ));
                }
                pending_root_alpha_override = Some(*alpha_state);
                (Ok((*bounds, None)), true)
            }
        };
        let atomic_compact_committed =
            atomic_stage_a_committed && atomic_root_c_compact_rows.is_some();
        let alpha_candidate = pending_root_alpha_override.is_some();
        let publication = match (applied_prune, spec_result.as_ref()) {
            (Some(plan), Ok((bounds, _cache))) => Some(finalize_root_spec_prune_publication(
                plan,
                spec_bounds_to_vec(bounds),
                pending_root_alpha_override.take(),
                atomic_compact_committed,
                atomic_binding_map_valid,
            )),
            _ => None,
        };
        let bounds_reconstruction_succeeded = publication
            .as_ref()
            .is_some_and(|publication| publication.bounds_reconstruction_succeeded);
        let mut merged_pruned_bounds = publication
            .as_ref()
            .and_then(|publication| publication.bounds.clone());
        // Before an atomic commit, a malformed compressed result may retry the
        // historical full-spec request. After a compact commit it may not cross
        // into another backend transaction: retain the complete independently
        // certified bootstrap pre-bound vector and discard any pending alpha.
        if publication
            .as_ref()
            .is_some_and(|publication| publication.retry_full_spec)
        {
            if retry_full_spec_after_compact_merge_failure(atomic_compact_committed) {
                debug!(
                    "Root spec pre-prune produced an invalid compact result; retrying the full spec request"
                );
                spec_result = run_spec_request(full_spec_mat);
                applied_prune = None;
            } else {
                debug!(
                    "Committed compact exact-C result failed reconstruction; retaining bootstrap reference without a second backend transaction"
                );
            }
        }
        root_alpha_override = publication
            .and_then(|publication| publication.alpha)
            .or(pending_root_alpha_override);
        if atomic_compact_committed {
            crate::execution_telemetry::record_exact_c_compact_commit(
                bounds_reconstruction_succeeded,
                atomic_binding_map_valid,
                alpha_candidate,
                root_alpha_override.is_some(),
            );
        }

        if let Some(plan) = applied_prune.filter(|_| spec_result.is_ok()) {
            crate::execution_telemetry::record_root_spec_prune_applied(
                objectives.len(),
                plan.active_indices.len(),
                objectives.len() - plan.active_indices.len(),
                false,
                exact_c_compressed_selected,
            );
        } else if exact_c_compressed_selected {
            crate::execution_telemetry::record_exact_c_compressed_selection_rolled_back(
                objectives.len(),
                atomic_evaluated_rows,
                exact_c_precertified_rows,
            );
        }

        // #post-c-survivor Stage B (dark/default-OFF): Stage A above remains
        // the exact historic full-C request. A bounds-only fast candidate is
        // identifiable by its successful bounds plus absent backward cache.
        // Only then, and only for the uncompressed disjunctive root, offer its
        // <=16 unresolved rows to one generic full-DAG Patch-CROWN backward.
        // The helper has no alpha-state input and publishes bounds+cache only
        // after all rows validate, so every decline/fault retains Stage A
        // byte-for-byte and cannot leak a partially tightened row.
        let post_c_survivor: Option<PostCSurvivorAccepted> = if verifier.config.root_post_c_survivor
            && applied_prune.is_none()
            && !atomic_stage_a_committed
        {
            spec_result
                .as_ref()
                .ok()
                .filter(|(_stage_a_bounds, stage_a_cache)| stage_a_cache.is_none())
                .and_then(|(stage_a_bounds, _stage_a_cache)| {
                    let plan = build_post_c_survivor_plan(
                        true,
                        conjunctive,
                        stage_a_bounds,
                        full_spec_mat,
                        thresholds,
                        input,
                        &bootstrap.initial_node_bounds,
                        std::time::Instant::now(),
                        global_deadline,
                    )?;
                    info!(
                        active_rows = plan.active_indices.len(),
                        total_rows = objectives.len(),
                        estimated_workspace_bytes = plan.estimated_workspace_bytes,
                        "Multi-objective: running bounded post-C survivor Patch-CROWN"
                    );
                    run_post_c_survivor_candidate(
                        graph,
                        input,
                        &bootstrap.initial_node_bounds,
                        engine,
                        plan,
                    )
                })
        } else {
            None
        };

        match spec_result {
            Ok((bounds, cache)) => {
                let stage_a_obj_bounds = if let Some(plan) = applied_prune {
                    // Populated above from this exact `bounds` result.
                    merged_pruned_bounds
                        .take()
                        .unwrap_or_else(|| plan.pre_bounds.clone())
                } else {
                    spec_bounds_to_vec(&bounds)
                };
                if let Some(post_c) = post_c_survivor.as_ref() {
                    info!(
                        active_rows = post_c.active_indices.len(),
                        total_rows = objectives.len(),
                        "Multi-objective: post-C survivor Patch-CROWN accepted atomically"
                    );
                }
                let stage_a_active_indices = applied_prune.map_or_else(
                    || (0..objectives.len()).collect(),
                    |plan| plan.active_indices.clone(),
                );
                let (mut obj_bounds, selected_cache, selected_active_indices) =
                    select_post_c_survivor_or_stage_a(
                        stage_a_obj_bounds,
                        cache,
                        stage_a_active_indices,
                        post_c_survivor,
                    );
                // DARK exact opt-in. This runs after every compressed/post-C
                // result has been restored to full row order, then authorizes
                // only one finite scalar lower bound. The typed request cannot
                // return a cache or fall through to CPU/forward-linear work.
                if root_critical_gpu_spec_enabled() {
                    let started_at = std::time::Instant::now();
                    let authority_deadline = match (root_deadline, global_deadline) {
                        (Some(root), Some(global)) => Some(root.min(global)),
                        (Some(root), None) => Some(root),
                        (None, global) => global,
                    };
                    match build_critical_gpu_spec_plan(
                        root_dense_head_stage_selected,
                        conjunctive,
                        verifier.config.crown_backward_layers,
                        &obj_bounds,
                        full_spec_mat,
                        thresholds,
                        started_at,
                        authority_deadline,
                    ) {
                        Err(reason) => {
                            emit_critical_gpu_spec_telemetry(
                                CriticalGpuSpecTelemetry::PlanRefused { reason },
                            );
                            info!(
                                status = "refused",
                                ?reason,
                                "Critical-row fresh-slope GPU root candidate"
                            );
                        }
                        Ok(plan) => {
                            let row = plan.row_index;
                            let routed = run_critical_gpu_spec_candidate(
                                graph,
                                input,
                                &bootstrap.initial_node_bounds,
                                engine,
                                &plan,
                                obj_bounds[row],
                            );
                            match routed.result {
                                Ok(accepted) => {
                                    let publication_refusal = apply_critical_gpu_spec_lower_only(
                                        &mut obj_bounds,
                                        row,
                                        &accepted,
                                        &plan,
                                    )
                                    .err();
                                    if let Some(reason) = publication_refusal {
                                        emit_critical_gpu_spec_telemetry(
                                            CriticalGpuSpecTelemetry::CandidateRefused {
                                                backend: routed.backend,
                                                row,
                                                reason,
                                            },
                                        );
                                        info!(
                                            status = "refused",
                                            backend = routed.backend.telemetry_name(),
                                            ?reason,
                                            row,
                                            elapsed_ms = started_at.elapsed().as_millis() as u64,
                                            "Critical-row fresh-slope GPU root candidate"
                                        );
                                    } else {
                                        emit_critical_gpu_spec_telemetry(
                                            CriticalGpuSpecTelemetry::Accepted {
                                                backend: routed.backend,
                                                row,
                                                accepted: &accepted,
                                            },
                                        );
                                        info!(
                                            status = "accepted",
                                            backend = routed.backend.telemetry_name(),
                                            row,
                                            elapsed_ms = started_at.elapsed().as_millis() as u64,
                                            historical_lower = accepted.historical_lower,
                                            candidate_lower = accepted.candidate_lower,
                                            merged_lower = accepted.merged_lower,
                                            lift =
                                                accepted.merged_lower - accepted.historical_lower,
                                            cache_published = false,
                                            "Critical-row fresh-slope GPU root candidate"
                                        );
                                    }
                                }
                                Err(reason) => {
                                    emit_critical_gpu_spec_telemetry(
                                        CriticalGpuSpecTelemetry::CandidateRefused {
                                            backend: routed.backend,
                                            row,
                                            reason,
                                        },
                                    );
                                    info!(
                                        status = "refused",
                                        backend = routed.backend.telemetry_name(),
                                        ?reason,
                                        row,
                                        elapsed_ms = started_at.elapsed().as_millis() as u64,
                                        "Critical-row fresh-slope GPU root candidate"
                                    );
                                }
                            }
                        }
                    }
                }
                root_spec_cache = selected_cache;
                root_spec_cache_active_indices = selected_active_indices;
                info!(
                    "Multi-objective: Using spec-guided CROWN ({} active / {} total objectives, cache_captured={})",
                    root_spec_cache_active_indices.len(),
                    obj_bounds.len(),
                    root_spec_cache.is_some(),
                );
                // Thread the deadline (#4321) on the historical route: this full
                // IBP forward over a deep conv DAG can itself overrun the verifier
                // timeout. A committed atomic Stage-A transaction already
                // authenticated the bootstrap output as its independent reference,
                // so reuse that certified enclosure instead of performing a second
                // root transaction against the possibly exhausted local deadline.
                let output = resolve_graph_root_output_after_atomic_stage_a(
                    graph,
                    bootstrap,
                    atomic_stage_a_committed,
                    || graph.propagate_ibp_with_engine_and_deadline(input, engine, root_deadline),
                )?;
                (output, obj_bounds)
            }
            Err(e) => {
                debug!(
                    "Spec-guided CROWN failed ({}), falling back to CROWN output bounds",
                    e
                );
                // This is still root-objective evaluation, so use the same
                // globally capped grace deadline as the spec request. In
                // particular, a retained phase checkpoint carries an expired
                // LOCAL warmup deadline; reusing it here would discard the
                // certified artifact despite a live outer verifier budget.
                let output = if super::super::is_finite_constrained_crown_refusal(&e) {
                    // An optional finite constrained implementation declined
                    // before publishing any coefficient state.
                    // Run only the graph's cooperative local IBP fallback; an
                    // expired root authority remains terminal and never mints
                    // fresh time for the fallback.
                    graph.propagate_ibp_with_engine_and_deadline(input, engine, root_deadline)?
                } else {
                    match compute_graph_root_output_bounds(
                        graph,
                        input,
                        &verifier.config,
                        engine,
                        bootstrap,
                        root_deadline,
                    ) {
                        Ok(output) => output,
                        Err(fallback_error)
                            if super::super::is_finite_constrained_crown_refusal(
                                &fallback_error,
                            ) =>
                        {
                            graph.propagate_ibp_with_engine_and_deadline(
                                input,
                                engine,
                                root_deadline,
                            )?
                        }
                        Err(fallback_error) => return Err(fallback_error),
                    }
                };
                let obj_bounds = match root_deadline {
                    Some(deadline) => {
                        crate::beta_crown::engine::graph::objectives::objective_bounds_multi_with_deadline(
                            &output,
                            objectives,
                            deadline,
                        )?
                    }
                    None => BetaCrownVerifier::objective_bounds_multi(&output, objectives)?,
                };
                (output, obj_bounds)
            }
        }
    } else {
        // The no-spec-matrix path is another root-objective evaluator and owns
        // the same globally capped grace as the spec request. This matters
        // after a phase checkpoint, whose local warmup deadline is expired by
        // definition but whose outer verifier authority remains live.
        let output = compute_graph_root_output_bounds(
            graph,
            input,
            &verifier.config,
            engine,
            bootstrap,
            root_deadline,
        )?;
        let obj_bounds = BetaCrownVerifier::objective_bounds_multi(&output, objectives)?;
        (output, obj_bounds)
    };

    // NY_SLACK_PROBE report: how many margin-units the accumulated f32 soundness
    // rounding (`lower_a_err` folded over the box at every node) removed from the
    // BINDING (min-margin) objective. If this is ≪ the ~0.3 gap to α,β-CROWN, the
    // gap is relaxation looseness (f64 cannot help), not precision slack.
    if crate::bounds::slack_probe_enabled() && root_intermediate_bounds_changed {
        // Best-of evaluates two independent certified candidates, so summing
        // their diagnostic fold penalties would not describe the selected
        // enclosure. Drain rather than emit a misleading "exact" number.
        let _ = crate::bounds::slack_probe_take();
        eprintln!(
            "[NY_SLACK_PROBE] unavailable for post-tightening best-of root evaluation (multiple sound candidates)"
        );
    } else if crate::bounds::slack_probe_enabled() {
        let slack = crate::bounds::slack_probe_take();
        let (mut worst_row, mut worst_lb) = (0usize, f32::INFINITY);
        for (r, &(l, _)) in initial_obj_bounds.iter().enumerate() {
            if l < worst_lb {
                worst_lb = l;
                worst_row = r;
            }
        }
        let binding_slack = slack.get(worst_row).copied().unwrap_or(0.0);
        let max_slack = slack.iter().copied().fold(0.0f64, f64::max);
        let sum_slack: f64 = slack.iter().sum();
        eprintln!(
            "[NY_SLACK_PROBE] objectives={} binding_row={worst_row} binding_margin_lb={worst_lb:.6} \
             binding_f32_slack={binding_slack:.6} max_row_slack={max_slack:.6} total_slack_all_rows={sum_slack:.6}",
            initial_obj_bounds.len()
        );
        eprintln!(
            "[NY_SLACK_PROBE] => f32 soundness rounding removed {binding_slack:.6} margin-units from the \
             binding objective; an exact/f64 backward could recover AT MOST that much of any gap to 0."
        );
    }

    Ok(RootObjectiveEvaluation {
        initial_output,
        initial_obj_bounds,
        root_spec_cache,
        root_alpha_override,
        root_spec_cache_active_indices,
    })
}

/// Root verdict authority for sign-normalized multi-objective rows.
///
/// Non-finite values are propagation failures, not proof certificates. In
/// particular, an overflowed `+Inf` lower bound must not satisfy the strict
/// stopping rule and close the property at the root.
fn root_objective_verified(lower: f32, upper: f32, threshold: f32) -> bool {
    root_prebound_certifies(lower, upper, threshold)
}

fn log_root_objective_bounds(initial_obj_bounds: &[(f32, f32)], thresholds: &[f32]) -> usize {
    let verified_count = initial_obj_bounds
        .iter()
        .zip(thresholds.iter())
        .filter(|((lower, upper), threshold)| root_objective_verified(*lower, *upper, **threshold))
        .count();

    for (idx, ((lower, upper), threshold)) in
        initial_obj_bounds.iter().zip(thresholds.iter()).enumerate()
    {
        info!(
            "Multi-objective obj[{}]: bounds=[{}, {}], threshold={}, verified={}",
            idx,
            lower,
            upper,
            threshold,
            root_objective_verified(*lower, *upper, *threshold)
        );
    }
    info!(
        "Multi-objective initial: {}/{} objectives already verified",
        verified_count,
        initial_obj_bounds.len()
    );

    verified_count
}

#[cfg(test)]
mod root_objective_authority_tests {
    use super::{log_root_objective_bounds, root_objective_verified};

    #[test]
    fn root_objective_verification_is_strict_and_rejects_non_finite_authority() {
        assert!(root_objective_verified(0.25, 1.0, 0.0));
        assert!(!root_objective_verified(0.0, 1.0, 0.0));
        assert!(!root_objective_verified(f32::INFINITY, f32::INFINITY, 0.0));
        assert!(!root_objective_verified(f32::NAN, 1.0, 0.0));
        assert!(!root_objective_verified(1.0, f32::INFINITY, 0.0));
        assert!(!root_objective_verified(1.0, f32::NAN, 0.0));
        assert!(!root_objective_verified(1.0, 0.5, 0.0));
        assert!(!root_objective_verified(1.0, 2.0, f32::INFINITY));
        assert!(!root_objective_verified(1.0, 2.0, f32::NAN));

        let bounds = [
            (f32::INFINITY, f32::INFINITY),
            (f32::NAN, 1.0),
            (0.0, 1.0),
            (0.25, 1.0),
        ];
        assert_eq!(
            log_root_objective_bounds(&bounds, &[0.0; 4]),
            1,
            "only the finite, strictly greater row may feed root closure"
        );
    }
}

fn maybe_finish_at_root(
    lifecycle: &mut GraphBabLifecycle,
    initial_output: BoundedTensor,
    initial_obj_bounds: &[(f32, f32)],
    thresholds: &[f32],
    conjunctive: bool,
    verified_count: usize,
) -> Option<BetaCrownResult> {
    let num_objectives = initial_obj_bounds.len();
    let initially_verified_all = verified_count == num_objectives;
    let initially_verified_any = verified_count > 0;

    if conjunctive && initially_verified_any {
        info!(
            "Multi-objective conjunctive: {}/{} objectives verified at root — property safe",
            verified_count, num_objectives
        );
        lifecycle.domains_explored = 1;
        lifecycle.domains_verified = 1;
        return Some(
            lifecycle.build_result_with_bounds(BabVerificationStatus::Verified, initial_output),
        );
    }
    if !conjunctive && initially_verified_all {
        lifecycle.domains_explored = 1;
        lifecycle.domains_verified = 1;
        return Some(
            lifecycle.build_result_with_bounds(BabVerificationStatus::Verified, initial_output),
        );
    }

    if conjunctive {
        let all_violated = initial_obj_bounds
            .iter()
            .zip(thresholds.iter())
            .all(|((_lower, upper), threshold)| *upper < *threshold);
        if all_violated {
            info!("Multi-objective conjunctive: ALL objectives conclusively violated at root");
            lifecycle.domains_explored = 1;
            return Some(lifecycle.build_result_with_bounds(
                BabVerificationStatus::Unknown {
                    reason:
                        "All objectives conclusively violated — conjunction may hold".to_string(),
                },
                initial_output,
            ));
        }
        return None;
    }

    for (idx, ((_lower, upper), threshold)) in
        initial_obj_bounds.iter().zip(thresholds.iter()).enumerate()
    {
        if *upper < *threshold {
            info!(
                "Multi-objective: objective {} is conclusively violated (upper={} < threshold={})",
                idx, upper, threshold
            );
            lifecycle.domains_explored = 1;
            return Some(lifecycle.build_result_with_bounds(
                BabVerificationStatus::Unknown {
                    reason: format!(
                        "Objective {} cannot be verified (upper {} < threshold {})",
                        idx, upper, threshold
                    ),
                },
                initial_output,
            ));
        }
    }

    None
}

fn attach_root_spec_cache(
    root_domain: &mut MultiObjectiveGraphBabDomain,
    root_spec_cache: Option<CachedLinearBounds>,
    active_indices: &[usize],
    num_objectives: usize,
) {
    let Some(multi_row_cache) = root_spec_cache else {
        return;
    };

    if let Some(per_objective_caches) =
        expand_root_spec_cache(&multi_row_cache, active_indices, num_objectives)
    {
        let captured_nodes = per_objective_caches
            .iter()
            .flatten()
            .next()
            .map_or(0, CachedLinearBounds::len);
        let captured_objectives = per_objective_caches.iter().flatten().count();
        if let Err(err) = root_domain.set_cached_las(per_objective_caches) {
            debug!("lA warm-start: failed to set cached_las on root: {err}");
        } else {
            info!(
                "lA warm-start: captured {} of {} per-objective cached_las on root domain across {} nodes",
                captured_objectives,
                num_objectives,
                captured_nodes,
            );
        }
    } else {
        debug!(
            "lA warm-start: cache row expansion returned None (index, linear-shape, or empty mismatch)"
        );
    }
}

/// Split compact cache rows and restore their full objective positions.  This
/// is deliberately total-order preserving and rejects duplicates/out-of-range
/// indices; a declined cache is only a performance loss, never a bound change.
fn expand_root_spec_cache(
    multi_row_cache: &CachedLinearBounds,
    active_indices: &[usize],
    num_objectives: usize,
) -> Option<Vec<Option<CachedLinearBounds>>> {
    if active_indices.is_empty() || num_objectives == 0 {
        return None;
    }
    let is_legacy_full_layout = active_indices.iter().copied().eq(0..num_objectives);
    // Legacy split_multi_row intentionally accepts "at least N" rows. Preserve
    // that exact behavior for the gate-off full layout, but require an exact
    // compact shape before mapping sparse rows: otherwise a mistakenly full
    // cache could silently attach its row 0 to active objective k.
    if !is_legacy_full_layout
        && (multi_row_cache
            .lower_a
            .values()
            .chain(multi_row_cache.upper_a.values())
            .any(|a| a.nrows() != active_indices.len())
            || multi_row_cache
                .lower_b
                .values()
                .chain(multi_row_cache.upper_b.values())
                .any(|b| b.len() != active_indices.len()))
    {
        return None;
    }

    let per_active = multi_row_cache.split_multi_row(active_indices.len())?;
    if per_active.len() != active_indices.len() {
        return None;
    }

    let mut seen = vec![false; num_objectives];
    let mut full: Vec<Option<CachedLinearBounds>> = vec![None; num_objectives];
    for (&full_idx, cache) in active_indices.iter().zip(per_active) {
        if full_idx >= num_objectives || seen[full_idx] {
            return None;
        }
        seen[full_idx] = true;
        full[full_idx] = Some(cache);
    }
    Some(full)
}

// ============================================================================
// DIAGNOSTIC-ONLY (NY_ROOT_WIDTH_PROBE=1) — cifar100 root looseness decomposition.
// Print-only. Not compiled out, but every effect is behind the env gate and never
// mutates the bootstrap or any verdict path. Branch: diag/cifar100-root-width.
// ============================================================================

fn probe_layer_kind(layer: &crate::Layer) -> &'static str {
    use crate::Layer;
    match layer {
        Layer::Conv2d(_) | Layer::Conv1d(_) => "conv",
        Layer::ConvTranspose2d(_) | Layer::ConvTranspose1d(_) => "convT",
        Layer::Linear(_) => "linear",
        Layer::ReLU(_) => "relu",
        Layer::BatchNorm(_) => "batchnorm",
        Layer::Add(_) => "add",
        Layer::Sub(_) => "sub",
        Layer::AveragePool(_) => "avgpool",
        Layer::MaxPool2d(_) => "maxpool",
        _ => "other",
    }
}

/// Width stats (max, mean) over `u_i - l_i` and numel for a BoundedTensor.
fn probe_width_stats(bt: &BoundedTensor) -> (f32, f32, usize) {
    let lo = bt.lower();
    let hi = bt.upper();
    let mut maxw = 0.0f32;
    let mut sumw = 0.0f64;
    let mut n = 0usize;
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        let w = u - l;
        if w.is_finite() {
            if w > maxw {
                maxw = w;
            }
            sumw += w as f64;
            n += 1;
        }
    }
    let mean = if n > 0 { (sumw / n as f64) as f32 } else { 0.0 };
    (maxw, mean, n)
}

/// Fraction of neurons with l<0<u (unstable — the only ReLUs whose relaxation has a gap).
fn probe_unstable_frac(bt: &BoundedTensor) -> (f32, usize, usize) {
    let lo = bt.lower();
    let hi = bt.upper();
    let mut unst = 0usize;
    let mut n = 0usize;
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        n += 1;
        if l < 0.0 && u > 0.0 {
            unst += 1;
        }
    }
    let frac = if n > 0 { unst as f32 / n as f32 } else { 0.0 };
    (frac, unst, n)
}

/// Build a copy of `map` with every non-input node's [l,u] shrunk toward its midpoint:
/// new half-width = `keep` * old half-width. `keep=1.0` is identity, `keep=0.0` collapses
/// to the midpoint. The graph input node is left untouched (the real property region).
fn probe_shrink_map(
    map: &std::collections::HashMap<String, BoundedTensor>,
    keep: f32,
    input_node: &str,
) -> std::collections::HashMap<String, BoundedTensor> {
    probe_shrink_filtered(map, keep, &|n| n != input_node)
}

/// Shrink toward midpoint only for nodes where `pred(name)` is true; others unchanged.
fn probe_shrink_filtered(
    map: &std::collections::HashMap<String, BoundedTensor>,
    keep: f32,
    pred: &dyn Fn(&str) -> bool,
) -> std::collections::HashMap<String, BoundedTensor> {
    map.iter()
        .map(|(name, bt)| {
            if !pred(name) {
                return (name.clone(), bt.clone());
            }
            let lo = bt.lower();
            let hi = bt.upper();
            let shape: Vec<usize> = lo.shape().to_vec();
            let new_lo_vec: Vec<f32> = lo
                .iter()
                .zip(hi.iter())
                .map(|(&l, &u)| {
                    // Bit-identical shrink center: f32::midpoint rounds differently at overflow/subnormal edges.
                    #[allow(clippy::manual_midpoint)]
                    let mid = (l + u) * 0.5f32;
                    let half = (u - l) * 0.5f32 * keep;
                    mid - half
                })
                .collect();
            let new_hi_vec: Vec<f32> = lo
                .iter()
                .zip(hi.iter())
                .map(|(&l, &u)| {
                    // Bit-identical shrink center: f32::midpoint rounds differently at overflow/subnormal edges.
                    #[allow(clippy::manual_midpoint)]
                    let mid = (l + u) * 0.5f32;
                    let half = (u - l) * 0.5f32 * keep;
                    mid + half
                })
                .collect();
            let new_lo = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), new_lo_vec);
            let new_hi = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), new_hi_vec);
            let shrunk = match (new_lo, new_hi) {
                (Ok(nl), Ok(nh)) => BoundedTensor::new(nl, nh).unwrap_or_else(|_| bt.clone()),
                _ => bt.clone(),
            };
            (name.clone(), shrunk)
        })
        .collect()
}

/// Min / mean of the per-objective margin LOWER bound + count verified (lower>threshold).
fn probe_margin_stats(out: &BoundedTensor, thresholds: &[f32]) -> (f32, f32, usize, usize) {
    let lo = out.lower();
    let vals: Vec<f32> = lo.iter().copied().collect();
    let mut min = f32::INFINITY;
    let mut sum = 0.0f64;
    let mut verified = 0usize;
    for (i, &v) in vals.iter().enumerate() {
        if v < min {
            min = v;
        }
        sum += v as f64;
        if v > thresholds.get(i).copied().unwrap_or(0.0) {
            verified += 1;
        }
    }
    let n = vals.len();
    let mean = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
    (min, mean, verified, n)
}

/// Run the SpecCrownRequest margin pass with a given intermediate-bound map (+ fixed root α).
fn probe_margin_with_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec: &ndarray::Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    alpha: Option<&crate::bounds::GraphAlphaState>,
    thresholds: &[f32],
) -> Option<(f32, f32, usize, usize)> {
    let out = SpecCrownRequest::new(graph, input, spec, engine)
        .node_bounds(node_bounds)
        .alpha_state_opt(alpha)
        .deadline_opt(None)
        .run()
        .ok()?;
    Some(probe_margin_stats(&out, thresholds))
}

#[allow(clippy::too_many_arguments)]
fn run_root_width_probe(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
    bootstrap: &GraphBabBootstrap,
    initial_obj_bounds: &[(f32, f32)],
) {
    let acrown = &bootstrap.initial_node_bounds;
    let alpha = bootstrap.root_alpha_state.as_ref();
    let out_name = if graph.output_name().is_empty() {
        graph
            .exec_order()
            .ok()
            .and_then(|o| o.last().cloned())
            .unwrap_or_default()
    } else {
        graph.output_name().to_string()
    };
    let input_node = graph
        .exec_order()
        .ok()
        .and_then(|o| o.first().cloned())
        .unwrap_or_default();

    // Plain IBP intermediate bounds for the A/B comparison.
    let ibp = graph.collect_node_bounds_with_engine(input, engine).ok();

    eprintln!(
        "[root-width] ===== ROOT LOOSENESS PROBE (out_node={out_name} input_node={input_node} n_obj={}) =====",
        objectives.len()
    );

    // ---- Per-node width profile in exec order (α-CROWN vs IBP intermediates) ----
    let order: Vec<String> = graph
        .exec_order()
        .map(|o| o.to_vec())
        .unwrap_or_else(|_| acrown.keys().cloned().collect());
    eprintln!("[root-width] --- per-node width (exec order): node kind numel | acrown[max mean] | ibp[max mean] | acrown/ibp ---");
    for name in &order {
        let Some(bt) = acrown.get(name) else { continue };
        let kind = graph
            .nodes
            .get(name)
            .map(|nd| probe_layer_kind(&nd.layer))
            .unwrap_or("?");
        let (amax, amean, an) = probe_width_stats(bt);
        let (imax, imean, ratio) = match ibp.as_ref().and_then(|m| m.get(name)) {
            Some(ib) => {
                let (im, imn, _) = probe_width_stats(ib);
                (im, imn, if imn > 0.0 { amean / imn } else { f32::NAN })
            }
            None => (f32::NAN, f32::NAN, f32::NAN),
        };
        eprintln!(
            "[root-width] node={name} kind={kind} numel={an} acrown_max={amax:.4} acrown_mean={amean:.4} ibp_max={imax:.4} ibp_mean={imean:.4} ratio={ratio:.3}"
        );
    }

    // ---- ReLU pre-activation unstable fractions (drives every triangle relaxation) ----
    eprintln!(
        "[root-width] --- ReLU pre-activation (input-node bounds): unstable fraction + width ---"
    );
    let mut tot_relu_neurons = 0usize;
    let mut tot_unstable_acrown = 0usize;
    let mut tot_unstable_ibp = 0usize;
    for name in &order {
        let Some(nd) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(nd.layer, crate::Layer::ReLU(_)) {
            continue;
        }
        let Some(in_name) = nd.inputs.first() else {
            continue;
        };
        let Some(pre) = acrown.get(in_name) else {
            continue;
        };
        let (afrac, aun, an) = probe_unstable_frac(pre);
        let (_amax, amean, _) = probe_width_stats(pre);
        tot_relu_neurons += an;
        tot_unstable_acrown += aun;
        let (ifrac, iun, imean) = match ibp.as_ref().and_then(|m| m.get(in_name)) {
            Some(ib) => {
                let (f, u, _) = probe_unstable_frac(ib);
                tot_unstable_ibp += u;
                let (_m, mn, _) = probe_width_stats(ib);
                (f, u, mn)
            }
            None => (f32::NAN, 0, f32::NAN),
        };
        eprintln!(
            "[root-width] relu={name} pre={in_name} numel={an} acrown_unstable={aun}({afrac:.3}) acrown_meanw={amean:.4} | ibp_unstable={iun}({ifrac:.3}) ibp_meanw={imean:.4}"
        );
    }
    eprintln!(
        "[root-width] RELU TOTALS: neurons={tot_relu_neurons} unstable_acrown={tot_unstable_acrown}({:.4}) unstable_ibp={tot_unstable_ibp}({:.4})",
        tot_unstable_acrown as f32 / tot_relu_neurons.max(1) as f32,
        tot_unstable_ibp as f32 / tot_relu_neurons.max(1) as f32,
    );

    // ---- Output-margin looseness decomposition ----
    let root_min = initial_obj_bounds
        .iter()
        .map(|(l, _)| *l)
        .fold(f32::INFINITY, f32::min);
    eprintln!(
        "[root-width] --- OUTPUT MARGIN DECOMPOSITION (min margin-lower; verify needs >0) ---"
    );
    eprintln!(
        "[root-width] real_root_margin_min={root_min:.5} (from compute_root_objective_bounds)"
    );

    // Pure IBP-concretized margin (no CROWN backward at all).
    if let Some(ibp_map) = ibp.as_ref() {
        if let Some(o) = ibp_map.get(&out_name) {
            let olo: Vec<f32> = o.lower().iter().copied().collect();
            let ohi: Vec<f32> = o.upper().iter().copied().collect();
            let mut min = f32::INFINITY;
            for obj in objectives {
                let mut lb = 0.0f32;
                for (j, &c) in obj.iter().enumerate() {
                    lb += if c >= 0.0 {
                        c * olo.get(j).copied().unwrap_or(0.0)
                    } else {
                        c * ohi.get(j).copied().unwrap_or(0.0)
                    };
                }
                if lb < min {
                    min = lb;
                }
            }
            eprintln!("[root-width] ibp_concretized_margin_min={min:.5} (IBP output box · objective, no backward)");
        }
    }

    let Some(spec) = build_spec_matrix(objectives) else {
        eprintln!("[root-width] build_spec_matrix failed; skipping CROWN-backward decomposition");
        return;
    };

    // Baseline: CROWN backward over α-CROWN intermediates + fixed root α.
    if let Some((mn, mean, ver, n)) =
        probe_margin_with_bounds(graph, input, &spec, engine, acrown, alpha, thresholds)
    {
        eprintln!("[root-width] CROWN[acrown_interm] margin_min={mn:.5} mean={mean:.5} verified={ver}/{n}");
    }
    // CROWN backward but with IBP intermediates (isolates: how much tighter intermediates buy).
    if let Some(ibp_map) = ibp.as_ref() {
        if let Some((mn, mean, ver, n)) =
            probe_margin_with_bounds(graph, input, &spec, engine, ibp_map, alpha, thresholds)
        {
            eprintln!("[root-width] CROWN[ibp_interm]    margin_min={mn:.5} mean={mean:.5} verified={ver}/{n}");
        }
    }
    // Artificially tightened intermediates: shrink each [l,u] toward its midpoint.
    // If margin barely moves => ReLU relaxation given these intermediates is the wall.
    // If margin jumps toward >0 => intermediate-bound looseness is the lever.
    for keep in [0.75f32, 0.5, 0.25, 0.1, 0.0] {
        let shrunk = probe_shrink_map(acrown, keep, &input_node);
        if let Some((mn, mean, ver, n)) =
            probe_margin_with_bounds(graph, input, &spec, engine, &shrunk, alpha, thresholds)
        {
            eprintln!(
                "[root-width] CROWN[shrink keep={keep:.2}] margin_min={mn:.5} mean={mean:.5} verified={ver}/{n}"
            );
        }
    }
    // TARGETED: isolate the FC head (the width-explosion layers) from the conv stack.
    // The pre-activation feeding the final ReLU (Relu_57) is Gemm_56. If shrinking ONLY
    // Gemm_56 closes most of the gap while shrinking everything-but-the-head barely moves,
    // the FC-head intermediate bounds are THE lever.
    let head_nodes = ["Gemm_56", "Relu_57", "Gemm_58"];
    let is_head = |n: &str| head_nodes.contains(&n);
    for keep in [0.5f32, 0.0] {
        let head_only = probe_shrink_filtered(acrown, keep, &|n| n == "Gemm_56");
        if let Some((mn, _mean, ver, n)) =
            probe_margin_with_bounds(graph, input, &spec, engine, &head_only, alpha, thresholds)
        {
            eprintln!("[root-width] CROWN[shrink Gemm_56-ONLY keep={keep:.2}] margin_min={mn:.5} verified={ver}/{n}");
        }
        let except_head = probe_shrink_filtered(acrown, keep, &|n| n != input_node && !is_head(n));
        if let Some((mn, _mean, ver, n)) =
            probe_margin_with_bounds(graph, input, &spec, engine, &except_head, alpha, thresholds)
        {
            eprintln!("[root-width] CROWN[shrink CONV-STACK-only(except head) keep={keep:.2}] margin_min={mn:.5} verified={ver}/{n}");
        }
    }
    eprintln!("[root-width] ===== END ROOT LOOSENESS PROBE =====");
}

/// Total finite box width Σ_j(u_j−l_j) and the finite-neuron count for a tensor.
fn probe_width_total(bt: &BoundedTensor) -> (f64, usize) {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (&l, &u) in bt.lower().iter().zip(bt.upper().iter()) {
        let w = (u - l) as f64;
        if w.is_finite() {
            sum += w;
            n += 1;
        }
    }
    (sum, n)
}

/// DIAGNOSTIC-ONLY (NY_ROOT_CROWN_INTERM_PROBE=1): compare, at the ROOT (no BaB
/// split), NY's FROZEN forward-linear pre-activation box width vs a sound
/// CROWN-backward pre-activation box width, per intermediate ReLU layer.
///
/// For every ReLU node, its pre-activation is the ReLU's single input node.
/// - (a) `fwd_linear_width` = Σ width of the frozen `initial_node_bounds[pre]`
///   (the forward-linear certified reference every BaB subdomain inherits);
/// - (b) `crown_heuristic_width` = Σ width of a sound CROWN BACKWARD from `pre`
///   to the input eps-box with heuristic α (`backward_input_relative_bounds_at_node`,
///   which folds the certified coeff error outward and refuses non-finite rows —
///   a valid enclosure) concretized via `LinearBounds::concretize_sound`;
/// - (c) `crown_optalpha_width` = the same sound backward with the warmup's
///   optimized α folded in (NY_ROOT_CROWN_INTERM_OPTALPHA≠0, when a warmup α exists).
///
/// `ratio_crown/fwd = (b)/(a)`: ≤0.5 ⇒ real headroom in the frozen intermediate
/// bounds (intersect the CROWN bounds in next); ~0.9–1.0 ⇒ the forward-linear
/// reference is already CROWN-tight (intermediate-bounds lever closed).
///
/// Print-only; never mutates any bound or verdict. See docs/ROOT_JOINT_INTERM_ALPHA_PLAN.md.
/// DIAGNOSTIC-ONLY (`NY_LPOPT_DUMP=<path>`): serialize the exact root state that
/// feeds every BaB subdomain, so the triangle-relaxation LP (`p*_LP`) can be
/// rebuilt off-line from NY's OWN bounds + relaxation.
///
/// Writes a plain-text file:
/// ```text
/// # ny lpopt dump v1
/// INPUT <n> <shape dims...>
/// L <l0> <l1> ...            # input eps-box lower, logical (row-major) order
/// U <u0> <u1> ...            # input eps-box upper
/// RELUMAP <k>
/// <relu_node_name> <pre_activation_node_name>   # one per ReLU, exec order
/// ...
/// NODE <name> <n> <shape dims...>
/// L <l0> ...                 # bootstrap.initial_node_bounds[name] lower
/// U <u0> ...
/// ...
/// ```
/// Floats use Rust's shortest round-trip `Display` (bit-exact reload of the f32
/// bounds NY's ReLU big-M / triangle relaxation actually casts to f64). Runs once
/// at the root, AFTER the optional CROWN-interm tighten, so it captures exactly the
/// frozen `initial_node_bounds` each subdomain inherits by pointer.
fn run_lpopt_dump(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bootstrap: &GraphBabBootstrap,
    path: &str,
) {
    use std::io::Write;
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[lpopt-dump] failed to create {path}: {e}");
            return;
        }
    };
    let mut w = std::io::BufWriter::new(file);
    let write_lu =
        |w: &mut std::io::BufWriter<std::fs::File>, bt: &BoundedTensor| -> std::io::Result<()> {
            let (lo, up) = bt.lower_upper();
            write!(w, "L")?;
            for v in lo.iter() {
                write!(w, " {v}")?;
            }
            writeln!(w)?;
            write!(w, "U")?;
            for v in up.iter() {
                write!(w, " {v}")?;
            }
            writeln!(w)
        };
    let res = (|| -> std::io::Result<()> {
        writeln!(w, "# ny lpopt dump v1")?;
        // Input eps-box.
        write!(w, "INPUT {}", input.len())?;
        for d in input.shape() {
            write!(w, " {d}")?;
        }
        writeln!(w)?;
        write_lu(&mut w, input)?;
        // ReLU -> pre-activation node-name map (exec order).
        let order = graph.exec_order().map(|o| o.to_vec()).unwrap_or_default();
        let relu_pairs: Vec<(String, String)> = order
            .iter()
            .filter_map(|name| {
                let node = graph.node(name)?;
                if !matches!(node.layer(), crate::Layer::ReLU(_)) {
                    return None;
                }
                let pre = node.inputs().first()?.clone();
                Some((name.clone(), pre))
            })
            .collect();
        writeln!(w, "RELUMAP {}", relu_pairs.len())?;
        for (relu, pre) in &relu_pairs {
            writeln!(w, "{relu} {pre}")?;
        }
        // Every node's frozen pre-activation box (keyed by node name).
        for (name, bt) in bootstrap.initial_node_bounds.iter() {
            write!(w, "NODE {name} {}", bt.len())?;
            for d in bt.shape() {
                write!(w, " {d}")?;
            }
            writeln!(w)?;
            write_lu(&mut w, bt)?;
        }
        w.flush()
    })();
    match res {
        Ok(()) => eprintln!(
            "[lpopt-dump] wrote {} nodes + input box to {path}",
            bootstrap.initial_node_bounds.len()
        ),
        Err(e) => eprintln!("[lpopt-dump] write error on {path}: {e}"),
    }
}

fn run_root_crown_interm_probe(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    bootstrap: &GraphBabBootstrap,
) {
    let Some(engine) = engine else {
        eprintln!("[root-crown-interm] no GPU engine (need the `ny vnncomp` GPU preset); skipping");
        return;
    };
    let acrown = &bootstrap.initial_node_bounds;
    // One-time Arc view of the frozen root map for the sound CROWN backward
    // (`backward_input_relative_bounds_at_node` reads the #cone-delta
    // increment 2 Arc-shared cache type). Diagnostic lane; values unchanged.
    let acrown_arc = build_initial_node_bounds_arc(acrown);
    let order: Vec<String> = match graph.exec_order() {
        Ok(o) => o.to_vec(),
        Err(_) => {
            eprintln!("[root-crown-interm] exec_order unavailable; skipping");
            return;
        }
    };
    // Bound the one-time probe cost: skip start-node seeds wider than this
    // (default 20000 = the whole cifar100 conv stack + head fits).
    let max_dim = std::env::var("NY_ROOT_CROWN_INTERM_MAXDIM")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(20000);
    // (c) optimized-α ceiling: build a root GraphDomainAlphaState from the warmup
    // α and fold it into the SAME sound backward. Default ON when a warmup α is
    // present; NY_ROOT_CROWN_INTERM_OPTALPHA=0 disables it (heuristic-α only).
    let want_optalpha = std::env::var("NY_ROOT_CROWN_INTERM_OPTALPHA")
        .ok()
        .as_deref()
        != Some("0");
    let domain_alpha: Option<GraphDomainAlphaState> = if want_optalpha {
        bootstrap.root_alpha_state.as_ref().map(|ra| {
            let arc = build_initial_node_bounds_arc(acrown);
            let hist = GraphSplitHistory::new();
            GraphDomainAlphaState::from_root_alpha_state(ra, graph, &arc, &hist, input)
        })
    } else {
        None
    };

    eprintln!(
        "[root-crown-interm] ===== ROOT CROWN-BACKWARD vs FROZEN FORWARD-LINEAR INTERM PROBE (max_dim={max_dim} optalpha={}) =====",
        domain_alpha.is_some()
    );
    let t0 = std::time::Instant::now();
    for name in &order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer, crate::Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs.first() else {
            continue;
        };
        let Some(ref_bt) = acrown.get(pre) else {
            continue;
        };
        let (fwd_sum, pre_dim) = probe_width_total(ref_bt);
        if pre_dim == 0 {
            continue;
        }
        if pre_dim > max_dim {
            eprintln!(
                "[root-crown-interm] relu={name} pre={pre} pre_dim={pre_dim} fwd_linear_width={fwd_sum:.4} crown_heuristic_width=SKIP(>max_dim) crown_optalpha_width=SKIP ratio_crown/fwd=SKIP"
            );
            continue;
        }
        // (b) heuristic-α sound CROWN backward from `pre` to the input eps-box.
        let crown_h = crate::beta_crown::engine::graph::propagation::batched::backward_input_relative_bounds_at_node(
            graph, pre, &acrown_arc, input, engine, None, None,
        )
        .map(|lb| probe_width_total(&lb.concretize_sound(input)).0);
        // (c) optimized-α sound CROWN backward (same lane, warmup α).
        let crown_o = domain_alpha.as_ref().and_then(|da| {
            crate::beta_crown::engine::graph::propagation::batched::backward_input_relative_bounds_at_node(
                graph, pre, &acrown_arc, input, engine, None, Some(da),
            )
            .map(|lb| probe_width_total(&lb.concretize_sound(input)).0)
        });
        let ch = crown_h
            .map(|w| format!("{w:.4}"))
            .unwrap_or_else(|| "REFUSED".into());
        let co = if domain_alpha.is_none() {
            "OFF".to_string()
        } else {
            crown_o
                .map(|w| format!("{w:.4}"))
                .unwrap_or_else(|| "REFUSED".into())
        };
        let ratio = crown_h
            .filter(|_| fwd_sum > 0.0)
            .map(|w| format!("{:.4}", w / fwd_sum))
            .unwrap_or_else(|| "NA".into());
        eprintln!(
            "[root-crown-interm] relu={name} pre={pre} pre_dim={pre_dim} fwd_linear_width={fwd_sum:.4} crown_heuristic_width={ch} crown_optalpha_width={co} ratio_crown/fwd={ratio}"
        );
    }
    eprintln!(
        "[root-crown-interm] ===== END ({:.1}s) =====",
        t0.elapsed().as_secs_f32()
    );
}

fn root_interm_cuda_factory_requested_from_raw(
    typed_requested: bool,
    raw: Option<&std::ffi::OsStr>,
) -> bool {
    match raw {
        None => typed_requested,
        Some(raw) => raw.to_str() == Some("1"),
    }
}

fn root_interm_cuda_factory_requested(config: &BetaCrownConfig) -> bool {
    root_interm_cuda_factory_requested_from_raw(
        config.root_interm_cuda_factory,
        std::env::var_os("NY_ROOT_INTERM_CUDA_FACTORY").as_deref(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RootIntermEngineRoute {
    Local,
    Factory,
    Unavailable,
}

/// Pick exactly one owner for an intermediate-bound invocation. A usable local
/// engine always owns the call (including a completed zero-result); the factory
/// is only eligible when no local attempt can start and its exact gate is armed.
pub(super) fn root_interm_engine_route(
    local_usable: bool,
    factory_requested: bool,
) -> RootIntermEngineRoute {
    if local_usable {
        RootIntermEngineRoute::Local
    } else if factory_requested {
        RootIntermEngineRoute::Factory
    } else {
        RootIntermEngineRoute::Unavailable
    }
}

/// Whether a sound sparse-CROWN backend can honor this lane's mandatory finite
/// deadline, either through its broad backend lease or its bounded-row
/// call-local entry point.
fn root_sparse_finite_deadline_capability_usable(
    provides_sound_gpu_crown: bool,
    honors_backend_deadline: bool,
    call_local_capacity: usize,
) -> bool {
    provides_sound_gpu_crown
        && (honors_backend_deadline
            || (1..=ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&call_local_capacity))
}

pub(super) fn provides_usable_sound_root_sparse_crown(engine: &dyn GemmEngine) -> bool {
    engine.as_gpu_crown_backward().is_some_and(|gpu| {
        root_sparse_finite_deadline_capability_usable(
            gpu.provides_sound_gpu_crown(),
            gpu.honors_crown_backward_deadline(),
            gpu.deadline_bounded_resnet_sound_max_rows(),
        )
    })
}

/// Default-dark first slice for one wide, demanded root intermediate target.
///
/// This policy is deliberately separate from [`RootSparseIntermCrownPolicy`]:
/// the production sparse selector and all of its defaults remain untouched,
/// while this experimental lane gets an independent one-target ranking policy
/// for the measured >=2,048-element wide class. The fixed caps are part of the
/// typed policy, not independently mutable environment reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RootWideDemandedIntermCrownPolicy {
    pub(super) min_dim: usize,
    pub(super) max_dim: usize,
    pub(super) max_rows: usize,
    pub(super) max_targets: usize,
    pub(super) max_preflights: usize,
    /// Caller-authorized ceiling for every simultaneously live device
    /// allocation owned by one typed intermediate-sweep request.
    pub(super) max_device_bytes: usize,
    /// Dispatch/publication authority. A synchronous extractor work unit is
    /// polled before and after but cannot be preempted midway.
    pub(super) max_secs: u64,
}

const ROOT_WIDE_DEMANDED_INTERM_MIN_DIM: usize = 2_048;
const ROOT_WIDE_DEMANDED_INTERM_MAX_DIM: usize = 32_768;
const ROOT_WIDE_DEMANDED_INTERM_MAX_ROWS: usize = 512;
const ROOT_WIDE_DEMANDED_INTERM_MAX_TARGETS: usize = 1;
const ROOT_WIDE_DEMANDED_INTERM_MAX_PREFLIGHTS: usize = 8;
const ROOT_WIDE_DEMANDED_INTERM_MAX_DEVICE_BYTES: usize = 512 * 1024 * 1024;
const ROOT_WIDE_DEMANDED_INTERM_MAX_SECS: u64 = 8;

fn resolve_root_wide_demanded_interm_crown_policy(
    enabled: bool,
) -> Option<RootWideDemandedIntermCrownPolicy> {
    enabled.then_some(RootWideDemandedIntermCrownPolicy {
        min_dim: ROOT_WIDE_DEMANDED_INTERM_MIN_DIM,
        max_dim: ROOT_WIDE_DEMANDED_INTERM_MAX_DIM,
        max_rows: ROOT_WIDE_DEMANDED_INTERM_MAX_ROWS,
        max_targets: ROOT_WIDE_DEMANDED_INTERM_MAX_TARGETS,
        max_preflights: ROOT_WIDE_DEMANDED_INTERM_MAX_PREFLIGHTS,
        max_device_bytes: ROOT_WIDE_DEMANDED_INTERM_MAX_DEVICE_BYTES,
        max_secs: ROOT_WIDE_DEMANDED_INTERM_MAX_SECS,
    })
}

pub(super) fn root_wide_demanded_interm_crown_policy() -> Option<RootWideDemandedIntermCrownPolicy>
{
    resolve_root_wide_demanded_interm_crown_policy(
        ny_levers::read(&ny_levers::decls::collection::ROOT_WIDE_DEMANDED_INTERM_CROWN)
            .value
            .as_bool(),
    )
}

/// Default-dark comprehensive GPU intermediate-sweep policy.
///
/// This is deliberately separate from the legacy one-target policy. When its
/// lever is armed it owns the root slot even on a clean decline, preventing a
/// comprehensive attempt from silently degrading into a different serial
/// verdict route. The backend supplies the device-specific row and memory
/// recommendation; these fields are only architecture-neutral absolute caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RootComprehensiveGpuIntermCrownPolicy {
    pub(super) min_dim: usize,
    pub(super) max_dim: usize,
    pub(super) max_rows_per_target: usize,
    pub(super) max_targets: usize,
    pub(super) max_device_bytes: usize,
    pub(super) max_secs: u64,
    /// Disjoint row windows to accumulate; 1 = the historical single sweep.
    pub(super) chunks: usize,
}

const ROOT_COMPREHENSIVE_GPU_INTERM_MIN_DIM: usize = 2_048;
const ROOT_COMPREHENSIVE_GPU_INTERM_MAX_DIM: usize = 32_768;
const ROOT_COMPREHENSIVE_GPU_INTERM_MAX_ROWS_PER_TARGET: usize = 32;
const ROOT_COMPREHENSIVE_GPU_INTERM_MAX_TARGETS: usize = 16;
const ROOT_COMPREHENSIVE_GPU_INTERM_MAX_DEVICE_BYTES: usize = 12 * 1024 * 1024 * 1024;
const ROOT_COMPREHENSIVE_GPU_INTERM_MAX_SECS: u64 = 20;

/// #comprehensive-rows-probe (measurement-only, `NY_ROOT_COMP_GPU_INTERM_ROWS`).
///
/// The comprehensive sweep is the ONLY mechanism measured to reach every eligible
/// root target atomically on the GPU, but it ships with a 32-row-per-target
/// absolute cap and the device profile retried down to 16 — i.e. 144 rows against
/// ~55,000 eligible neurons, 0.26% coverage. Partial coverage is already known not
/// to convert (`top-3 = 69% of width, still verified 0/99`), so the open design
/// question is the SCALING LAW: is the 1.4 GB peak observed at 144 rows mostly
/// fixed overhead, or marginal per-row cost? Those imply completely different
/// designs (one big sweep vs. row-chunked accumulation), and the answer cannot be
/// inferred from a single data point.
///
/// This override only RAISES the absolute per-target row cap for measurement. It
/// cannot make a bound unsound: the backend still validates the whole typed
/// request, still honours `max_device_bytes` and the deadline, and the host still
/// commits shrink-only. An over-large request is refused by the backend, which is
/// exactly the signal being measured. Absent/malformed leaves the shipped cap.
fn root_comprehensive_gpu_interm_rows_override() -> Option<usize> {
    ny_levers::read(&ny_levers::decls::comprehensive_rows::ROOT_COMP_GPU_INTERM_ROWS)
        .value
        .as_u64()
        .and_then(|rows| usize::try_from(rows).ok())
        .filter(|rows| *rows > 0)
}

/// #comprehensive-rows-probe: measurement override for the phase's local
/// authority slice.
///
/// With row-chunking the sweep now trades TIME for coverage, and at the official
/// 100 s budget the 20 s slice is what binds: only four ~4.3 s chunks fit, giving
/// 512 rows/target and a root census of 82/99 on `idx_2132` (0/99 unchunked). The
/// open question is whether more slice converts that into a root PROOF (99/99),
/// which decides whether the phase deserves a larger share of the budget.
///
/// Raising it cannot make a bound unsound — every chunk is still atomic,
/// deadline-bounded and shrink-only — it can only spend more of the budget here
/// and less elsewhere, which is exactly the trade being measured.
fn root_comprehensive_gpu_interm_secs_override() -> Option<u64> {
    ny_levers::read(&ny_levers::decls::comprehensive_rows::ROOT_COMP_GPU_INTERM_SECS)
        .value
        .as_u64()
        .filter(|secs| *secs > 0)
}

fn resolve_root_comprehensive_gpu_interm_crown_policy(
    enabled: bool,
    chunks: usize,
) -> Option<RootComprehensiveGpuIntermCrownPolicy> {
    enabled.then_some(RootComprehensiveGpuIntermCrownPolicy {
        chunks: chunks.max(1),
        min_dim: ROOT_COMPREHENSIVE_GPU_INTERM_MIN_DIM,
        max_dim: ROOT_COMPREHENSIVE_GPU_INTERM_MAX_DIM,
        max_rows_per_target: root_comprehensive_gpu_interm_rows_override()
            .unwrap_or(ROOT_COMPREHENSIVE_GPU_INTERM_MAX_ROWS_PER_TARGET),
        max_targets: ROOT_COMPREHENSIVE_GPU_INTERM_MAX_TARGETS,
        max_device_bytes: ROOT_COMPREHENSIVE_GPU_INTERM_MAX_DEVICE_BYTES,
        max_secs: root_comprehensive_gpu_interm_secs_override()
            .unwrap_or(ROOT_COMPREHENSIVE_GPU_INTERM_MAX_SECS),
    })
}

/// DELIVERY SEAM (`measured_gate_delivery.rs`): the scored entry point exports
/// exactly one `NY_*` variable, so an env-only gate cannot fire in competition
/// however well it measured. The typed preset key is therefore the primary
/// source and the env lever is an explicit force-on override for A/B/rollback.
pub(super) fn root_comprehensive_gpu_interm_crown_policy(
    config: &BetaCrownConfig,
) -> Option<RootComprehensiveGpuIntermCrownPolicy> {
    let lever_on =
        ny_levers::read(&ny_levers::decls::collection::ROOT_COMPREHENSIVE_GPU_INTERM_CROWN)
            .value
            .as_bool();
    // Consult the GOVERNED lever, not a raw env read, so this stays inside the
    // registry's chokepoint. Only an EXPLICIT environment value overrides the
    // preset: the declaration's own default is `1`, so testing the value alone
    // would let the shipped default silently beat the preset key and the typed
    // delivery would be dead on arrival — the exact failure this seam exists to
    // prevent. `Source::LegacyEnv` is the only source that means "a human asked
    // for this"; `LegacyEnvRejected` deliberately does NOT qualify, because a
    // malformed override is a kill switch rather than permission to reveal a
    // preset.
    let lever = ny_levers::read(&ny_levers::decls::comprehensive_rows::INTERM_ROW_CHUNKS);
    let chunks = if matches!(lever.source, ny_levers::Source::LegacyEnv) {
        usize::try_from(lever.value.as_u64().unwrap_or(1)).unwrap_or(1)
    } else {
        config.root_comprehensive_gpu_interm_chunks
    };
    resolve_root_comprehensive_gpu_interm_crown_policy(
        config.root_comprehensive_gpu_interm || lever_on,
        chunks,
    )
}

/// Default-dark phase-resident dense plus comprehensive sweep policy.
///
/// These are architecture-neutral absolute caps. The retained backend's live
/// resource policy and exact request preflight remain the final capacity
/// authority. `max_dense_rows` is a complete-census cap: exceeding it refuses
/// the whole route rather than selecting a dense prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RootPhaseResidentCrownPolicy {
    pub(super) min_comprehensive_dim: usize,
    pub(super) max_comprehensive_dim: usize,
    pub(super) max_comprehensive_rows_per_target: usize,
    pub(super) max_comprehensive_targets: usize,
    pub(super) max_dense_rows: usize,
    pub(super) max_device_bytes: usize,
    pub(super) max_secs: u64,
}

const ROOT_PHASE_RESIDENT_MIN_COMPREHENSIVE_DIM: usize = 2_048;
const ROOT_PHASE_RESIDENT_MAX_COMPREHENSIVE_DIM: usize = 32_768;
const ROOT_PHASE_RESIDENT_MAX_COMPREHENSIVE_ROWS_PER_TARGET: usize = 32;
const ROOT_PHASE_RESIDENT_MAX_COMPREHENSIVE_TARGETS: usize = 16;
const ROOT_PHASE_RESIDENT_MAX_DENSE_ROWS: usize = 512;
const ROOT_PHASE_RESIDENT_MAX_DEVICE_BYTES: usize = 8 * 1024 * 1024 * 1024;
const ROOT_PHASE_RESIDENT_MAX_SECS: u64 = 20;

fn resolve_root_phase_resident_crown_policy(enabled: bool) -> Option<RootPhaseResidentCrownPolicy> {
    enabled.then_some(RootPhaseResidentCrownPolicy {
        min_comprehensive_dim: ROOT_PHASE_RESIDENT_MIN_COMPREHENSIVE_DIM,
        max_comprehensive_dim: ROOT_PHASE_RESIDENT_MAX_COMPREHENSIVE_DIM,
        max_comprehensive_rows_per_target: ROOT_PHASE_RESIDENT_MAX_COMPREHENSIVE_ROWS_PER_TARGET,
        max_comprehensive_targets: ROOT_PHASE_RESIDENT_MAX_COMPREHENSIVE_TARGETS,
        max_dense_rows: ROOT_PHASE_RESIDENT_MAX_DENSE_ROWS,
        max_device_bytes: ROOT_PHASE_RESIDENT_MAX_DEVICE_BYTES,
        max_secs: ROOT_PHASE_RESIDENT_MAX_SECS,
    })
}

pub(super) fn root_phase_resident_crown_policy() -> Option<RootPhaseResidentCrownPolicy> {
    resolve_root_phase_resident_crown_policy(
        ny_levers::read(&ny_levers::decls::collection::ROOT_PHASE_RESIDENT_CROWN)
            .value
            .as_bool(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RootSparseIntermCrownPolicy {
    pub(super) max_dim: usize,
    pub(super) max_rows: usize,
    pub(super) max_targets: usize,
    pub(super) max_secs: u64,
}

const ROOT_SPARSE_INTERM_ABS_MAX_DIM: usize = 8_192;
const ROOT_SPARSE_INTERM_ABS_MAX_ROWS: usize = 512;
const ROOT_SPARSE_INTERM_ABS_MAX_TARGETS: usize = 4;
const ROOT_SPARSE_INTERM_ABS_MAX_SECS: u64 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootSparseGateEnv<'a> {
    Absent,
    Unicode(&'a str),
    NonUnicode,
}

fn root_sparse_gate_env(raw: Option<&std::ffi::OsStr>) -> RootSparseGateEnv<'_> {
    match raw {
        None => RootSparseGateEnv::Absent,
        Some(value) => value
            .to_str()
            .map_or(RootSparseGateEnv::NonUnicode, RootSparseGateEnv::Unicode),
    }
}

/// Resolve the typed sparse-row policy plus sealed environment overrides without
/// touching process-global state. Only an exact raw `"1"` enables and an exact
/// raw `"0"` disables. Any other present value fails closed, even when the typed
/// config is enabled. Explicit zero caps fail closed at the selector/deadline
/// boundary.
#[allow(clippy::too_many_arguments)]
fn resolve_root_sparse_interm_crown_policy(
    config: &BetaCrownConfig,
    gate_env: RootSparseGateEnv<'_>,
    max_dim_env: Option<&str>,
    max_rows_env: Option<&str>,
    max_targets_env: Option<&str>,
    max_secs_env: Option<&str>,
) -> Option<RootSparseIntermCrownPolicy> {
    let enabled = match gate_env {
        RootSparseGateEnv::Absent => config.root_sparse_interm_crown,
        RootSparseGateEnv::Unicode("1") => true,
        RootSparseGateEnv::Unicode(_) | RootSparseGateEnv::NonUnicode => false,
    };
    if !enabled {
        return None;
    }
    Some(RootSparseIntermCrownPolicy {
        max_dim: max_dim_env
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(config.root_sparse_interm_crown_max_dim)
            .min(ROOT_SPARSE_INTERM_ABS_MAX_DIM),
        max_rows: max_rows_env
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(config.root_sparse_interm_crown_max_rows)
            .min(ROOT_SPARSE_INTERM_ABS_MAX_ROWS),
        max_targets: max_targets_env
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(config.root_sparse_interm_crown_max_targets)
            .min(ROOT_SPARSE_INTERM_ABS_MAX_TARGETS),
        max_secs: max_secs_env
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(config.root_sparse_interm_crown_max_secs)
            .min(ROOT_SPARSE_INTERM_ABS_MAX_SECS),
    })
}

pub(super) fn root_sparse_interm_crown_policy_from_env(
    config: &BetaCrownConfig,
) -> Option<RootSparseIntermCrownPolicy> {
    let gate = std::env::var_os("NY_ROOT_SPARSE_INTERM_CROWN");
    let max_dim = std::env::var("NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM").ok();
    let max_rows = std::env::var("NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS").ok();
    let max_targets = std::env::var("NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS").ok();
    let max_secs = std::env::var("NY_ROOT_SPARSE_INTERM_CROWN_SECS").ok();
    resolve_root_sparse_interm_crown_policy(
        config,
        root_sparse_gate_env(gate.as_deref()),
        max_dim.as_deref(),
        max_rows.as_deref(),
        max_targets.as_deref(),
        max_secs.as_deref(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RootCrownIntermSelection {
    /// Structural production scope: Linear/Gemm producers immediately feeding
    /// a ReLU. This finds cifar100's head without relying on ONNX node names.
    DenseHead,
    /// Legacy diagnostic scope retained for environment-driven experiments.
    All,
    /// Legacy measured high-Δ node-name set.
    Preset,
    /// Legacy explicit ReLU node-name list.
    Explicit(std::collections::HashSet<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RootCrownIntermPolicy {
    pub(super) selection: RootCrownIntermSelection,
    pub(super) max_dim: usize,
    pub(super) max_secs: u64,
}

/// Resolve typed production policy plus legacy environment overrides without
/// reading process-global state. Kept pure so default-off and kill-switch
/// semantics are directly testable.
fn resolve_root_crown_interm_policy(
    config: &BetaCrownConfig,
    gate_env: Option<&str>,
    layers_env: Option<&str>,
    max_dim_env: Option<&str>,
    max_secs_env: Option<&str>,
) -> Option<RootCrownIntermPolicy> {
    let env_forced_on = gate_env.is_some_and(|value| value.trim() == "1");
    let enabled = match gate_env.map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        // Unknown inherited values are not interpreted as a force-on; the typed
        // preset remains authoritative.
        _ => config.root_crown_interm_dense_head,
    };
    if !enabled {
        return None;
    }

    let selection = match layers_env.map(str::trim) {
        Some(value) if value.is_empty() || value.eq_ignore_ascii_case("all") => {
            RootCrownIntermSelection::All
        }
        Some(value)
            if value.eq_ignore_ascii_case("dense-head")
                || value.eq_ignore_ascii_case("dense_head")
                || value.eq_ignore_ascii_case("head") =>
        {
            RootCrownIntermSelection::DenseHead
        }
        Some(value) if value.eq_ignore_ascii_case("preset") => RootCrownIntermSelection::Preset,
        Some(value) => RootCrownIntermSelection::Explicit(
            value
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .collect(),
        ),
        // Preserve the old `NY_ROOT_CROWN_INTERM=1` no-layers behavior (`all`)
        // when it alone force-enables an otherwise-off config. A typed preset
        // selects the production dense-head scope.
        None if env_forced_on && !config.root_crown_interm_dense_head => {
            RootCrownIntermSelection::All
        }
        None => RootCrownIntermSelection::DenseHead,
    };
    let max_dim = max_dim_env
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or({
            if env_forced_on && !config.root_crown_interm_dense_head {
                // Preserve the original env-only experiment contract. Before
                // the typed production preset existed, force-on without an
                // explicit MAXDIM admitted the complete CIFAR conv stack.
                20_000
            } else {
                config.root_crown_interm_max_dim
            }
        });
    let max_secs = max_secs_env
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(config.root_crown_interm_max_secs);
    Some(RootCrownIntermPolicy {
        selection,
        max_dim,
        max_secs,
    })
}

pub(super) fn root_crown_interm_policy_from_env(
    config: &BetaCrownConfig,
) -> Option<RootCrownIntermPolicy> {
    let gate = std::env::var("NY_ROOT_CROWN_INTERM").ok();
    let layers = std::env::var("NY_ROOT_CROWN_INTERM_LAYERS").ok();
    let max_dim = std::env::var("NY_ROOT_CROWN_INTERM_MAXDIM").ok();
    let max_secs = std::env::var("NY_ROOT_CROWN_INTERM_SECS").ok();
    resolve_root_crown_interm_policy(
        config,
        gate.as_deref(),
        layers.as_deref(),
        max_dim.as_deref(),
        max_secs.as_deref(),
    )
}

/// Give the pass at most its configured cap and at most half of the remaining
/// global verifier budget. Expired/zero/tiny slices skip the pass, so the root
/// objective and BaB retain their original sound bounds and remaining wall time.
pub(super) fn bounded_root_crown_interm_deadline(
    now: std::time::Instant,
    global_deadline: Option<std::time::Instant>,
    max_secs: u64,
) -> Option<std::time::Instant> {
    if max_secs == 0 {
        return None;
    }
    let cap = std::time::Duration::from_secs(max_secs);
    // NOTE: the 0.5 share, not `max_secs`, is what bounds the row-chunked sweep
    // (max_secs 40 and 60 both yield exactly 5 chunks at the official budget).
    // Raising it was tried and DELIBERATELY NOT KEPT: it only moves wall clock
    // from BaB to this phase, and budget reallocation on cifar100 is already
    // measured worthless (5 timeout rows x {30s,50s} = 0/10, and 2025 scoring has
    // add_time_bonus=False). Coverage has to come from a cheaper sweep, not from
    // a bigger slice of the same budget.
    let slice = match global_deadline {
        Some(global) => {
            let remaining = global.checked_duration_since(now)?;
            cap.min(remaining.mul_f32(0.5))
        }
        None => cap,
    };
    // Below this floor a graph walk cannot usefully start. Skipping is the
    // fail-closed outcome: the existing sound box remains untouched.
    if slice < std::time::Duration::from_millis(100) {
        return None;
    }
    now.checked_add(slice)
}

/// High-Δ preset (`NY_ROOT_CROWN_INTERM_LAYERS=preset`): the deep ReLU
/// pre-activations the probe measured as materially looser than the CROWN
/// backward (Relu_13/19/25/31/57) plus the cheap 2048-wide deep blocks
/// (Relu_39/45/51). Names are matched against the ReLU node names in exec order.
const ROOT_CROWN_INTERM_PRESET: &[&str] = &[
    "Relu_13", "Relu_19", "Relu_25", "Relu_31", "Relu_39", "Relu_45", "Relu_51", "Relu_57",
];

/// SHRINK-ONLY per-element intersect of a frozen forward-linear reference box
/// `ref_bt` with a sound CROWN box `crown` (flat `[num_outputs]`, same element
/// count, iterated in matching logical order). For each element:
///   `l_new = max(l_fwd, l_crown)`, `u_new = min(u_fwd, u_crown)`.
/// FAIL-CLOSED and NEVER-WIDEN: if the CROWN endpoints are non-finite/inverted, or
/// the intersect would invert (`l_new > u_new`, disjoint enclosures ⇒ upstream
/// bug or an infeasible domain), the reference element is kept verbatim. The
/// result is therefore always `l_new ∈ [l_fwd, u_fwd]`, `u_new ∈ [l_fwd, u_fwd]`,
/// `l_new ≤ u_new` — a sound tightening of `ref_bt` that never drops a real point
/// (both inputs enclose the reachable set, so does their intersection).
///
/// Returns `(tightened, n_tightened_elems)`, or `None` on element-count mismatch
/// or if the rebuilt tensor is rejected (⇒ caller keeps the reference).
fn shrink_only_intersect(
    ref_bt: &BoundedTensor,
    crown: &BoundedTensor,
) -> Option<(BoundedTensor, usize)> {
    if ref_bt.len() != crown.len() {
        return None;
    }
    let (mut nl, mut nu) = ref_bt.clone().into_parts();
    let mut n_tightened = 0usize;
    for ((l, u), (&cl, &cu)) in nl
        .iter_mut()
        .zip(nu.iter_mut())
        .zip(crown.lower().iter().zip(crown.upper().iter()))
    {
        // Fail-closed: only tighten from a finite, valid CROWN endpoint.
        if !cl.is_finite() || !cu.is_finite() || cl > cu {
            continue;
        }
        let lf = *l;
        let uf = *u;
        let cand_l = lf.max(cl);
        let cand_u = uf.min(cu);
        // Never invert (disjoint boxes ⇒ keep the reference; never widen).
        if cand_l <= cand_u {
            // Shrink-only invariant (both hold by construction of max/min).
            debug_assert!(
                cand_l >= lf && cand_u <= uf,
                "root-crown-interm widened a bound"
            );
            if cand_l > lf || cand_u < uf {
                n_tightened += 1;
            }
            *l = cand_l;
            *u = cand_u;
        }
    }
    BoundedTensor::new_allow_infinite(nl, nu)
        .ok()
        .map(|bt| (bt, n_tightened))
}

/// Publish one computed dense intermediate box only while the pass still owns
/// live wall-clock authority. The check is intentionally adjacent to the sole
/// map mutation so a candidate computed before the deadline cannot be inserted
/// after the deadline has already expired.
fn publish_root_crown_interm_bound_at(
    bounds: &mut std::collections::HashMap<String, BoundedTensor>,
    pre: &str,
    tightened: BoundedTensor,
    pass_deadline: std::time::Instant,
    now: std::time::Instant,
) -> bool {
    if now >= pass_deadline {
        return false;
    }
    bounds.insert(pre.to_owned(), tightened);
    true
}

/// Count crossing-ReLU (unstable) pre-activation neurons across all ReLU nodes: a
/// neuron is unstable when its pre-activation box straddles 0 (`l < 0 < u`). Used
/// only for the before/after diagnostic log (baseline root count ≈ 1008 / 970).
fn count_unstable_relu_preacts(
    graph: &GraphNetwork,
    bounds: &std::collections::HashMap<String, BoundedTensor>,
) -> usize {
    let order = match graph.exec_order() {
        Ok(o) => o.to_vec(),
        Err(_) => return 0,
    };
    let mut n = 0usize;
    for name in &order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer, crate::Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs.first() else {
            continue;
        };
        let Some(bt) = bounds.get(pre) else {
            continue;
        };
        for (&l, &u) in bt.lower().iter().zip(bt.upper().iter()) {
            if l < 0.0 && u > 0.0 {
                n += 1;
            }
        }
    }
    n
}

/// `#root-crown-interm`: at the ROOT, tighten the frozen forward-linear
/// pre-activation bounds by SHRINK-ONLY intersecting a sound heuristic-α CROWN
/// backward box into each selected intermediate ReLU pre-activation of
/// `bootstrap.initial_node_bounds`. Runs ONCE at the root; the typed production
/// scope is dense-head only, and the tightened map is then Arc-shared to every
/// BaB subdomain by the caller.
///
/// Two-phase (compute-then-apply) so every CROWN backward reads the ORIGINAL frozen
/// bounds (matching the probe's measured widths; avoids in-pass feedback and the
/// immutable/mutable borrow overlap): phase 1 computes each node's CROWN box from
/// the frozen cache; phase 2 shrink-only intersects them in. SOUNDNESS: see
/// `shrink_only_intersect` — never widens, never drops a real point, fail-closed.
/// Returns the number of elements whose lower or upper endpoint actually shrank;
/// zero is the clean signal for no target/no improvement/deadline/refusal.
pub(super) fn run_root_crown_interm_tighten(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: &dyn GemmEngine,
    bootstrap: &mut GraphBabBootstrap,
    policy: &RootCrownIntermPolicy,
    pass_deadline: std::time::Instant,
) -> usize {
    let order: Vec<String> = match graph.exec_order() {
        Ok(o) => o.to_vec(),
        Err(_) => {
            eprintln!(
                "[root-crown-interm-tighten] exec_order unavailable; skipping (bounds unchanged)"
            );
            return 0;
        }
    };
    let max_dim = policy.max_dim;
    let dense_head_targets: std::collections::HashSet<String> = graph
        .fc_head_preactivation_targets(&order)
        .into_iter()
        .collect();
    let want = |relu_name: &str, pre_name: &str| -> bool {
        match &policy.selection {
            RootCrownIntermSelection::DenseHead => dense_head_targets.contains(pre_name),
            RootCrownIntermSelection::All => true,
            RootCrownIntermSelection::Preset => ROOT_CROWN_INTERM_PRESET.contains(&relu_name),
            RootCrownIntermSelection::Explicit(names) => names.contains(relu_name),
        }
    };
    let selection_label = match &policy.selection {
        RootCrownIntermSelection::DenseHead => "dense-head".to_string(),
        RootCrownIntermSelection::All => "all".to_string(),
        RootCrownIntermSelection::Preset => "preset".to_string(),
        RootCrownIntermSelection::Explicit(names) => {
            let mut names: Vec<&str> = names.iter().map(String::as_str).collect();
            names.sort_unstable();
            names.join(",")
        }
    };

    let unstable_before = count_unstable_relu_preacts(graph, &bootstrap.initial_node_bounds);
    eprintln!(
        "[root-crown-interm-tighten] ===== ROOT SHRINK-ONLY CROWN INTERSECT (layers={selection_label} max_dim={max_dim} budget={:.3}s) unstable_before={unstable_before} =====",
        pass_deadline.saturating_duration_since(std::time::Instant::now()).as_secs_f32(),
    );
    let t0 = std::time::Instant::now();

    // Phase 1 — compute every selected node's sound CROWN box from the ORIGINAL
    // frozen bounds (immutable borrow only).
    let mut computed: Vec<(String, BoundedTensor)> = Vec::new();
    {
        let acrown = &bootstrap.initial_node_bounds;
        // One-time Arc view for the sound CROWN backward (see probe above).
        let acrown_arc = build_initial_node_bounds_arc(acrown);
        for name in &order {
            if std::time::Instant::now() >= pass_deadline {
                eprintln!(
                    "[root-crown-interm-tighten] deadline reached before next target -> keep remaining fwd_linear bounds"
                );
                break;
            }
            let Some(node) = graph.nodes.get(name) else {
                continue;
            };
            if !matches!(node.layer, crate::Layer::ReLU(_)) {
                continue;
            }
            let Some(pre) = node.inputs.first() else {
                continue;
            };
            if !want(name, pre) {
                continue;
            }
            let Some(ref_bt) = acrown.get(pre) else {
                continue;
            };
            let pre_dim = ref_bt.len();
            if pre_dim == 0 {
                continue;
            }
            if pre_dim > max_dim {
                eprintln!(
                    "[root-crown-interm-tighten] relu={name} pre={pre} pre_dim={pre_dim} SKIP(>max_dim)"
                );
                continue;
            }
            // Sound heuristic-α CROWN backward (α=None), certified error folded
            // outward; refuses non-finite ⇒ Option. Concretize soundly over the box.
            //
            // #node-timing: this pass is the ONLY thing measured to move the root
            // objective census (verified 0/99 -> 3/99) and it costs ~349 s for
            // nine nodes against a ~20 s affordable slice. Four hypotheses about
            // WHERE that time goes have already been falsified, so record the
            // per-node cost instead of inferring it — a uniform ~39 s/node and a
            // one-node-dominates profile call for completely different fixes.
            let node_t0 = std::time::Instant::now();
            let crown = crate::beta_crown::engine::graph::propagation::batched::backward_input_relative_bounds_at_node(
                graph, pre, &acrown_arc, input, engine, Some(pass_deadline), None,
            )
            .map(|lb| lb.concretize_sound(input));
            if crate::phase_telemetry::phase_telemetry_enabled() {
                eprintln!(
                    "[interm-node] pre={pre} pre_dim={pre_dim} elapsed={:.2}s got={}",
                    node_t0.elapsed().as_secs_f64(),
                    crown.is_some(),
                );
            }
            match crown {
                Some(cbox) => computed.push((pre.clone(), cbox)),
                None if std::time::Instant::now() >= pass_deadline => {
                    eprintln!(
                        "[root-crown-interm-tighten] relu={name} pre={pre} deadline/refusal -> keep fwd_linear and stop"
                    );
                    break;
                }
                None => eprintln!(
                    "[root-crown-interm-tighten] relu={name} pre={pre} pre_dim={pre_dim} CROWN REFUSED -> keep fwd_linear"
                ),
            }
        }
    }

    // Phase 2 — shrink-only intersect each CROWN box into the frozen map.
    let mut nodes_tightened = 0usize;
    let mut elements_tightened = 0usize;
    let mut total_fwd_w = 0.0f64;
    let mut total_new_w = 0.0f64;
    for (pre, cbox) in &computed {
        if std::time::Instant::now() >= pass_deadline {
            eprintln!(
                "[root-crown-interm-tighten] deadline reached before phase-2 target -> keep \
                 remaining fwd_linear bounds"
            );
            break;
        }
        let Some(ref_bt) = bootstrap.initial_node_bounds.get(pre) else {
            continue;
        };
        let (fwd_w, _) = probe_width_total(ref_bt);
        match shrink_only_intersect(ref_bt, cbox) {
            Some((tightened, n_elems)) => {
                let (new_w, _) = probe_width_total(&tightened);
                if !publish_root_crown_interm_bound_at(
                    &mut bootstrap.initial_node_bounds,
                    pre,
                    tightened,
                    pass_deadline,
                    std::time::Instant::now(),
                ) {
                    eprintln!(
                        "[root-crown-interm-tighten] deadline reached before phase-2 publication \
                         for pre={pre} -> keep this and remaining fwd_linear bounds"
                    );
                    break;
                }
                total_fwd_w += fwd_w;
                total_new_w += new_w;
                if n_elems > 0 {
                    nodes_tightened += 1;
                    elements_tightened += n_elems;
                }
                eprintln!(
                    "[root-crown-interm-tighten] pre={pre} fwd_linear_width={fwd_w:.4} -> intersected_width={new_w:.4} tightened_elems={n_elems}"
                );
            }
            None => eprintln!(
                "[root-crown-interm-tighten] pre={pre} shape/len mismatch or rebuild rejected -> keep fwd_linear"
            ),
        }
    }

    let unstable_after = count_unstable_relu_preacts(graph, &bootstrap.initial_node_bounds);
    let elapsed = t0.elapsed().as_secs_f32();
    let reduction = total_fwd_w - total_new_w;
    let unstable_delta = (unstable_before as i64) - (unstable_after as i64);
    let n_computed = computed.len();
    eprintln!(
        "[root-crown-interm-tighten] ===== END ({elapsed:.1}s) nodes_tightened={nodes_tightened}/{n_computed} total_width {total_fwd_w:.4} -> {total_new_w:.4} (reduction {reduction:.4}) unstable {unstable_before} -> {unstable_after} (Δ {unstable_delta}) ====="
    );
    elements_tightened
}

#[cfg(test)]
mod root_crown_interm_tests {
    use super::{
        bounded_root_crown_interm_deadline, comprehensive_gpu_or_legacy_wide,
        phase_resident_or_comprehensive, publish_root_crown_interm_bound_at,
        resolve_root_comprehensive_gpu_interm_crown_policy, resolve_root_crown_interm_policy,
        resolve_root_phase_resident_crown_policy, resolve_root_sparse_interm_crown_policy,
        resolve_root_wide_demanded_interm_crown_policy,
        root_interm_cuda_factory_requested_from_raw, root_interm_engine_route,
        root_intermediate_tightening_changed, root_sparse_finite_deadline_capability_usable,
        root_sparse_gate_env, shrink_only_intersect, RootCrownIntermSelection,
        RootIntermEngineRoute, RootSparseGateEnv,
    };
    use crate::BetaCrownConfig;
    use ny_tensor::BoundedTensor;
    use std::cell::Cell;
    use std::collections::HashMap;

    fn bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        use ndarray::{ArrayD, IxDyn};
        BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn typed_root_crown_interm_is_default_off_and_dense_head_when_armed() {
        let default = BetaCrownConfig::default();
        assert!(
            resolve_root_crown_interm_policy(&default, None, None, None, None).is_none(),
            "missing preset/env must leave the new root pass off"
        );

        let armed = BetaCrownConfig {
            root_crown_interm_dense_head: true,
            root_crown_interm_max_secs: 3,
            root_crown_interm_max_dim: 321,
            ..BetaCrownConfig::default()
        };
        let policy = resolve_root_crown_interm_policy(&armed, None, None, None, None).unwrap();
        assert_eq!(policy.selection, RootCrownIntermSelection::DenseHead);
        assert_eq!(policy.max_secs, 3);
        assert_eq!(policy.max_dim, 321);
    }

    #[test]
    fn every_root_intermediate_producer_invalidates_pre_tightening_alpha() {
        let dense = |elements, targets| super::super::root_phases::PhaseOutput {
            dense_head_stage_selected: true,
            tightened_elements: elements,
            tightened_targets: targets,
        };
        assert!(!root_intermediate_tightening_changed(
            dense(0, 0),
            None,
            0,
            0,
            0
        ));
        assert!(!root_intermediate_tightening_changed(
            dense(0, 0),
            Some(0),
            0,
            0,
            0
        ));

        // Each producer must independently invalidate state built for the old
        // frozen boxes. The comprehensive-only case is the regression: its
        // result used to be omitted at four downstream consumers.
        assert!(root_intermediate_tightening_changed(
            dense(1, 0),
            None,
            0,
            0,
            0
        ));
        assert!(root_intermediate_tightening_changed(
            dense(0, 1),
            None,
            0,
            0,
            0
        ));
        assert!(root_intermediate_tightening_changed(
            dense(0, 0),
            Some(1),
            0,
            0,
            0
        ));
        assert!(root_intermediate_tightening_changed(
            dense(0, 0),
            None,
            1,
            0,
            0
        ));
        assert!(root_intermediate_tightening_changed(
            dense(0, 0),
            None,
            0,
            1,
            0
        ));
        assert!(root_intermediate_tightening_changed(
            dense(0, 0),
            None,
            0,
            0,
            1
        ));
    }

    #[test]
    fn root_intermediate_change_summary_is_wired_to_every_consumer() {
        // Normalize line endings before matching: several needles below span
        // lines with `\n`, but `.rs` checks out CRLF under core.autocrlf, so on
        // Windows they matched 0 times and this wiring gate failed against a
        // file whose content was exactly right.
        let source = include_str!("root.rs").replace("\r\n", "\n");
        let summary_declaration = [
            "let root_intermediate_bounds_changed = ",
            "root_intermediate_tightening_changed(",
        ]
        .concat();
        let objective_argument = [
            "deadline,\n        root_intermediate_bounds_changed,\n",
            "        root_dense_head_stage_selected,",
        ]
        .concat();
        let stale_alpha_guard = ["if root_intermediate_", "bounds_changed {"].concat();
        let root_box_summary = [
            "let root_boxes_tightened = root_intermediate_",
            "bounds_changed;",
        ]
        .concat();
        let resident_owner = [
            "phase_resident_or_",
            "comprehensive(root_phase_resident_crown_policy(), ||",
        ]
        .concat();
        let deferred_execution = [
            "RootTightenPhase::PhaseResidentCrown\n",
            "            .run_phase_resident_in_place(",
        ]
        .concat();
        let preaccept_fallback = ["if resident.permits_legacy_", "dense_fallback() {"].concat();
        let phase_output_summary = [
            "root_intermediate_tightening_changed(\n",
            "        dense_head_out,",
        ]
        .concat();

        assert_eq!(source.matches(&summary_declaration).count(), 1);
        assert_eq!(source.matches(&objective_argument).count(), 1);
        // Two root-domain alpha builders and the root-objective request must
        // all consume the same summary rather than reconstructing it.
        assert_eq!(source.matches(&stale_alpha_guard).count(), 3);
        assert_eq!(source.matches(&root_box_summary).count(), 1);
        assert_eq!(source.matches(&resident_owner).count(), 1);
        assert_eq!(source.matches(&deferred_execution).count(), 1);
        assert_eq!(source.matches(&preaccept_fallback).count(), 1);
        assert_eq!(source.matches(&phase_output_summary).count(), 1);
    }

    #[test]
    fn root_interm_cuda_factory_resolver_inherits_typed_and_accepts_exactly_one() {
        assert!(!root_interm_cuda_factory_requested_from_raw(false, None));
        assert!(root_interm_cuda_factory_requested_from_raw(true, None));
        assert!(root_interm_cuda_factory_requested_from_raw(
            false,
            Some(std::ffi::OsStr::new("1"))
        ));
        assert!(root_interm_cuda_factory_requested_from_raw(
            true,
            Some(std::ffi::OsStr::new("1"))
        ));
        for malformed in ["", "0", "01", " 1", "1 ", "true", "yes"] {
            for typed_requested in [false, true] {
                assert!(
                    !root_interm_cuda_factory_requested_from_raw(
                        typed_requested,
                        Some(std::ffi::OsStr::new(malformed))
                    ),
                    "present malformed value {malformed:?} must force off"
                );
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn root_interm_cuda_factory_non_unicode_override_forces_off() {
        use std::os::unix::ffi::OsStringExt;

        let non_unicode = std::ffi::OsString::from_vec(vec![0xff]);
        assert!(!root_interm_cuda_factory_requested_from_raw(
            true,
            Some(non_unicode.as_os_str())
        ));
    }

    #[test]
    fn root_interm_local_route_owns_completed_zero_without_factory_retry() {
        assert_eq!(
            root_interm_engine_route(true, true),
            RootIntermEngineRoute::Local,
            "a usable local route remains the sole owner even when the factory gate is armed"
        );
        assert_eq!(
            root_interm_engine_route(false, true),
            RootIntermEngineRoute::Factory
        );
        assert_eq!(
            root_interm_engine_route(false, false),
            RootIntermEngineRoute::Unavailable
        );
    }

    #[test]
    fn root_sparse_finite_deadline_capability_controls_factory_fallback() {
        let max = ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS;
        let cases = [
            (false, false, 0, false),
            (false, false, 1, false),
            (false, false, max, false),
            (false, false, max + 1, false),
            (false, true, 0, false),
            (false, true, 1, false),
            (false, true, max, false),
            (false, true, max + 1, false),
            (true, false, 0, false),
            (true, false, 1, true),
            (true, false, max, true),
            (true, false, max + 1, false),
            (true, true, 0, true),
            (true, true, 1, true),
            (true, true, max, true),
            (true, true, max + 1, true),
        ];
        for (sound, honors_deadline, call_local_capacity, expected) in cases {
            assert_eq!(
                root_sparse_finite_deadline_capability_usable(
                    sound,
                    honors_deadline,
                    call_local_capacity,
                ),
                expected,
                "sound={sound} honors_deadline={honors_deadline} \
                 call_local_capacity={call_local_capacity}"
            );
        }

        let deadline_ineligible_local =
            root_sparse_finite_deadline_capability_usable(true, false, 0);
        assert_eq!(
            root_interm_engine_route(deadline_ineligible_local, true),
            RootIntermEngineRoute::Factory,
            "an armed factory must own the call when the local sound engine cannot honor the \
             finite deadline"
        );
    }

    #[test]
    fn typed_root_sparse_interm_is_default_off_bounded_and_killable() {
        let default = BetaCrownConfig::default();
        assert!(resolve_root_sparse_interm_crown_policy(
            &default,
            RootSparseGateEnv::Absent,
            None,
            None,
            None,
            None,
        )
        .is_none());

        let armed = BetaCrownConfig {
            root_sparse_interm_crown: true,
            root_sparse_interm_crown_max_secs: 3,
            root_sparse_interm_crown_max_dim: 4_096,
            root_sparse_interm_crown_max_rows: 96,
            root_sparse_interm_crown_max_targets: 2,
            ..BetaCrownConfig::default()
        };
        assert!(resolve_root_sparse_interm_crown_policy(
            &armed,
            RootSparseGateEnv::Unicode("0"),
            None,
            None,
            None,
            None,
        )
        .is_none());
        let policy = resolve_root_sparse_interm_crown_policy(
            &armed,
            RootSparseGateEnv::Absent,
            Some("2048"),
            Some("32"),
            Some("1"),
            Some("4"),
        )
        .unwrap();
        assert_eq!(policy.max_dim, 2_048);
        assert_eq!(policy.max_rows, 32);
        assert_eq!(policy.max_targets, 1);
        assert_eq!(policy.max_secs, 4);

        let forced = resolve_root_sparse_interm_crown_policy(
            &default,
            RootSparseGateEnv::Unicode("1"),
            Some("0"),
            Some("0"),
            Some("0"),
            Some("0"),
        )
        .unwrap();
        assert_eq!(forced.max_dim, 0);
        assert_eq!(forced.max_rows, 0);
        assert_eq!(forced.max_targets, 0);
        assert_eq!(forced.max_secs, 0);

        let clamped = resolve_root_sparse_interm_crown_policy(
            &default,
            RootSparseGateEnv::Unicode("1"),
            Some("999999"),
            Some("999999"),
            Some("999999"),
            Some("999999"),
        )
        .unwrap();
        assert_eq!(clamped.max_dim, 8_192);
        assert_eq!(clamped.max_rows, 512);
        assert_eq!(clamped.max_targets, 4);
        assert_eq!(clamped.max_secs, 8);

        for malformed in [" 1", "1 ", "true", "yes", "01", ""] {
            assert!(
                resolve_root_sparse_interm_crown_policy(
                    &armed,
                    RootSparseGateEnv::Unicode(malformed),
                    None,
                    None,
                    None,
                    None,
                )
                .is_none(),
                "present malformed gate {malformed:?} must disable even a typed-on config"
            );
        }
        assert!(
            resolve_root_sparse_interm_crown_policy(
                &armed,
                RootSparseGateEnv::NonUnicode,
                None,
                None,
                None,
                None,
            )
            .is_none(),
            "a present non-Unicode gate must disable even a typed-on config"
        );
        assert_eq!(
            root_sparse_gate_env(None),
            RootSparseGateEnv::Absent,
            "an absent environment gate must defer to typed config"
        );
    }

    #[test]
    fn typed_root_wide_demanded_interm_is_default_off_and_strictly_bounded() {
        assert!(resolve_root_wide_demanded_interm_crown_policy(false).is_none());
        let policy = resolve_root_wide_demanded_interm_crown_policy(true).unwrap();
        assert_eq!(policy.min_dim, 2_048);
        assert_eq!(policy.max_dim, 32_768);
        assert_eq!(policy.max_rows, 512);
        assert_eq!(policy.max_targets, 1);
        assert_eq!(policy.max_preflights, 8);
        assert_eq!(policy.max_device_bytes, 512 * 1024 * 1024);
        assert_eq!(policy.max_secs, 8);
    }

    #[test]
    fn comprehensive_gpu_interm_is_separate_dark_and_bounded_below_track_vram_cap() {
        assert!(resolve_root_comprehensive_gpu_interm_crown_policy(false, 1).is_none());
        let policy = resolve_root_comprehensive_gpu_interm_crown_policy(true, 1).unwrap();
        assert_eq!(policy.min_dim, 2_048);
        assert_eq!(policy.max_dim, 32_768);
        assert_eq!(policy.max_rows_per_target, 32);
        assert_eq!(policy.max_targets, 16);
        assert_eq!(policy.max_device_bytes, 12 * 1024 * 1024 * 1024);
        assert_eq!(policy.max_secs, 20);
    }

    #[test]
    fn phase_resident_crown_is_separate_dark_and_strictly_bounded() {
        assert!(resolve_root_phase_resident_crown_policy(false).is_none());
        let policy = resolve_root_phase_resident_crown_policy(true).unwrap();
        assert_eq!(policy.min_comprehensive_dim, 2_048);
        assert_eq!(policy.max_comprehensive_dim, 32_768);
        assert_eq!(policy.max_comprehensive_rows_per_target, 32);
        assert_eq!(policy.max_comprehensive_targets, 16);
        assert_eq!(policy.max_dense_rows, 512);
        assert_eq!(policy.max_device_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(policy.max_secs, 20);
    }

    #[test]
    fn resident_ownership_is_resolved_without_running_the_old_comprehensive_route() {
        let calls = Cell::new(0usize);
        let policy = resolve_root_phase_resident_crown_policy(true).unwrap();
        let (owner, comprehensive) = phase_resident_or_comprehensive(Some(policy), || {
            calls.set(calls.get() + 1);
            Some(7)
        });
        assert_eq!(owner, Some(policy));
        assert_eq!(comprehensive, Some(0));
        assert_eq!(calls.get(), 0);

        let (owner, comprehensive) = phase_resident_or_comprehensive(None, || {
            calls.set(calls.get() + 1);
            Some(7)
        });
        assert_eq!(owner, None);
        assert_eq!(comprehensive, Some(7));
        assert_eq!(
            calls.get(),
            1,
            "the gate-off legacy route runs exactly once"
        );
    }

    #[test]
    fn comprehensive_ownership_never_calls_the_legacy_wide_route() {
        let calls = Cell::new(0usize);
        let owned = comprehensive_gpu_or_legacy_wide(Some(0), || {
            calls.set(calls.get() + 1);
            99
        });
        assert_eq!(owned, 0, "an owned clean decline remains the phase result");
        assert_eq!(calls.get(), 0, "legacy backend route must stay untouched");

        let unarmed = comprehensive_gpu_or_legacy_wide(None, || {
            calls.set(calls.get() + 1);
            7
        });
        assert_eq!(unarmed, 7);
        assert_eq!(
            calls.get(),
            1,
            "legacy route runs exactly once when unarmed"
        );
    }

    #[test]
    fn root_crown_interm_env_retains_force_and_selection_overrides() {
        let armed = BetaCrownConfig {
            root_crown_interm_dense_head: true,
            ..BetaCrownConfig::default()
        };
        assert!(
            resolve_root_crown_interm_policy(&armed, Some("0"), None, None, None).is_none(),
            "NY_ROOT_CROWN_INTERM=0 must remain a production kill switch"
        );

        let off = BetaCrownConfig::default();
        let legacy =
            resolve_root_crown_interm_policy(&off, Some("1"), None, Some("42"), Some("7")).unwrap();
        assert_eq!(legacy.selection, RootCrownIntermSelection::All);
        assert_eq!(legacy.max_dim, 42);
        assert_eq!(legacy.max_secs, 7);

        let legacy_implicit =
            resolve_root_crown_interm_policy(&off, Some("1"), None, None, None).unwrap();
        assert_eq!(legacy_implicit.selection, RootCrownIntermSelection::All);
        assert_eq!(legacy_implicit.max_dim, 20_000);

        let explicit = resolve_root_crown_interm_policy(
            &armed,
            Some("1"),
            Some("Relu_57, custom_relu"),
            None,
            None,
        )
        .unwrap();
        let RootCrownIntermSelection::Explicit(names) = explicit.selection else {
            panic!("explicit env layer list must remain supported")
        };
        assert!(names.contains("Relu_57"));
        assert!(names.contains("custom_relu"));
    }

    #[test]
    fn root_crown_interm_deadline_is_capped_and_failclosed() {
        let now = std::time::Instant::now();
        assert_eq!(bounded_root_crown_interm_deadline(now, None, 0), None);
        assert_eq!(
            bounded_root_crown_interm_deadline(now, Some(now), 2),
            None,
            "expired global deadline must not start the pass"
        );
        assert_eq!(
            bounded_root_crown_interm_deadline(
                now,
                Some(now + std::time::Duration::from_secs(1)),
                2,
            ),
            Some(now + std::time::Duration::from_millis(500)),
            "pass gets at most half the remaining global budget"
        );
        assert_eq!(
            bounded_root_crown_interm_deadline(now, None, 2),
            Some(now + std::time::Duration::from_secs(2)),
            "without a global deadline the typed cap remains authoritative"
        );
    }

    #[test]
    fn root_crown_interm_never_publishes_a_late_phase_two_candidate() {
        let original = bt(&[-5.0], &[5.0]);
        let tightened = bt(&[-2.0], &[2.0]);
        let mut bounds = HashMap::from([("pre".to_owned(), original.clone())]);
        let expired = std::time::Instant::now();

        assert!(!publish_root_crown_interm_bound_at(
            &mut bounds,
            "pre",
            tightened.clone(),
            expired,
            expired,
        ));
        assert_eq!(bounds["pre"].lower(), original.lower());
        assert_eq!(bounds["pre"].upper(), original.upper());

        let checked_at = std::time::Instant::now();
        assert!(publish_root_crown_interm_bound_at(
            &mut bounds,
            "pre",
            tightened.clone(),
            checked_at + std::time::Duration::from_secs(1),
            checked_at,
        ));
        assert_eq!(bounds["pre"].lower(), tightened.lower());
        assert_eq!(bounds["pre"].upper(), tightened.upper());
    }

    /// Core soundness contract: the intersect is SHRINK-ONLY — every output element
    /// satisfies l_fwd ≤ l_new ≤ u_new ≤ u_fwd (never widens, never inverts).
    #[test]
    fn shrink_only_intersect_never_widens() {
        let fwd = bt(&[-5.0, -5.0, -1.0, 0.0], &[5.0, 5.0, 3.0, 10.0]);
        // CROWN box: tighter on some elems, looser on others (must be ignored when looser).
        let crown = bt(&[-2.0, -9.0, 2.0, 1.0], &[2.0, 9.0, 4.0, 6.0]);
        let (out, n) = shrink_only_intersect(&fwd, &crown).unwrap();
        let (fl, fu) = fwd.lower_upper();
        let (ol, ou) = out.lower_upper();
        for i in 0..4 {
            // never widened past the forward reference
            assert!(ol[i] >= fl[i] - 0.0, "elem {i} lower widened");
            assert!(ou[i] <= fu[i] + 0.0, "elem {i} upper widened");
            assert!(ol[i] <= ou[i], "elem {i} inverted");
        }
        // elem0: [-5,5]∩[-2,2] = [-2,2] tightened both sides
        assert_eq!(ol[0], -2.0);
        assert_eq!(ou[0], 2.0);
        // elem1: crown [-9,9] looser => keep forward [-5,5]
        assert_eq!(ol[1], -5.0);
        assert_eq!(ou[1], 5.0);
        // elem2: [-1,3]∩[2,4] = [2,3] lower tightened
        assert_eq!(ol[2], 2.0);
        assert_eq!(ou[2], 3.0);
        // elem3: [0,10]∩[1,6] = [1,6]
        assert_eq!(ol[3], 1.0);
        assert_eq!(ou[3], 6.0);
        assert_eq!(n, 3); // elems 0,2,3 moved; elem1 unchanged
    }

    /// Fail-closed: non-finite CROWN endpoints keep the forward reference verbatim.
    #[test]
    fn shrink_only_intersect_failclosed_on_nonfinite() {
        let fwd = bt(&[-5.0, -5.0], &[5.0, 5.0]);
        let crown = bt(&[f32::NEG_INFINITY, 1.0], &[f32::INFINITY, 2.0]);
        let (out, n) = shrink_only_intersect(&fwd, &crown).unwrap();
        let (ol, ou) = out.lower_upper();
        // elem0: crown non-finite => keep forward
        assert_eq!(ol[0], -5.0);
        assert_eq!(ou[0], 5.0);
        // elem1: finite crown [1,2] tightens
        assert_eq!(ol[1], 1.0);
        assert_eq!(ou[1], 2.0);
        assert_eq!(n, 1);
    }

    /// Fail-closed: disjoint boxes keep the forward reference (never widen/invert).
    #[test]
    fn shrink_only_intersect_failclosed_on_disjoint() {
        let fwd = bt(&[-5.0], &[-1.0]);
        let crown = bt(&[2.0], &[4.0]); // disjoint from [-5,-1]
        let (out, n) = shrink_only_intersect(&fwd, &crown).unwrap();
        let (ol, ou) = out.lower_upper();
        assert_eq!(ol[0], -5.0);
        assert_eq!(ou[0], -1.0);
        assert_eq!(n, 0);
    }

    /// Element-count mismatch => None (caller keeps the reference).
    #[test]
    fn shrink_only_intersect_len_mismatch_is_none() {
        let fwd = bt(&[-5.0, 1.0], &[5.0, 2.0]);
        let crown = bt(&[0.0], &[1.0]);
        assert!(shrink_only_intersect(&fwd, &crown).is_none());
    }

    /// The experimental row-conditional lane was published without its
    /// implementation/configuration module. Keep the production root free of
    /// dangling hooks until that lane returns as one complete, reviewed change.
    #[test]
    fn unavailable_experimental_lane_has_no_root_hook() {
        let source = include_str!("root.rs");
        for unavailable_symbol in [
            ["row_conditional", "_invprop_enabled"].concat(),
            ["run_row_conditional", "_invprop"].concat(),
            ["row_conditional", "_invprop_pass"].concat(),
            ["apply_row_conditional", "_outcome"].concat(),
        ] {
            assert!(
                !source.contains(&unavailable_symbol),
                "unavailable experimental hook '{unavailable_symbol}' must not break main"
            );
        }
    }
}
