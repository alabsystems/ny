// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default-dark paired transport of root warmup α into the first BaB children.
//!
//! The established tightened-box heuristic state is always evaluated first.
//! An explicitly expanded warmup state may become the next-domain continuation
//! state only when its worst active lower margin strictly improves. Published
//! lower-bound authority is deliberately separate: lower bounds are the maximum
//! of two independently sound lower certificates, while every upper endpoint is
//! retained byte-for-byte from H as the behavioral baseline. No β-derived child
//! upper endpoint is treated as a proof certificate.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::{GemmEngine, GpuCrownBackward};
use ny_tensor::BoundedTensor;

use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};

const SELECTIVE_ROOT_ALPHA_ENV: &str = "NY_SELECTIVE_ROOT_ALPHA";
pub(super) const SELECTIVE_ROOT_ALPHA_AUTHORITY_RESERVE: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SelectiveRootAlphaGate {
    Disabled,
    Enabled,
}

impl SelectiveRootAlphaGate {
    fn from_raw(raw: Option<&str>) -> Self {
        if raw == Some("1") {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }

    pub(super) fn from_env() -> Self {
        match std::env::var(SELECTIVE_ROOT_ALPHA_ENV) {
            Ok(raw) => Self::from_raw(Some(&raw)),
            Err(std::env::VarError::NotPresent | std::env::VarError::NotUnicode(_)) => {
                Self::Disabled
            }
        }
    }

    pub(super) const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Provenance of the continuation state installed beside published child bounds.
///
/// This is intentionally *not* a claim that one optimizer state produced both
/// endpoints. A W win publishes `max(H_lower, W_lower)` with H's unchanged
/// upper endpoint, then carries W's independently sound node enclosure plus
/// β/α warm starts into the next sound recomputation. Every state-dependent
/// linear-form cache is invalidated first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) enum ChildContinuationStateProvenance {
    Established,
    SelectiveExpandedWarmup,
}

impl ChildContinuationStateProvenance {
    pub(in crate::beta_crown::engine::graph) const fn invalidates_all_cached_las(self) -> bool {
        matches!(self, Self::SelectiveExpandedWarmup)
    }
}

pub(in crate::beta_crown::engine::graph) fn apply_child_continuation_state_provenance(
    domain: &mut crate::beta_crown::domain::MultiObjectiveGraphBabDomain,
    provenance: ChildContinuationStateProvenance,
) {
    if provenance.invalidates_all_cached_las() {
        domain.cached_las.fill(None);
        domain.clear_per_disjunct_alphas();
    }
}

/// Install an evaluated arm's continuation state.
///
/// Published objective bounds are installed separately by `update_bounds`;
/// this function has no authority over them. `node_bounds` is the selected
/// arm's independently sound enclosure, while β and α are admissible warm
/// starts for future recomputation. In particular, a selective W state first
/// clears every lA/per-disjunct cache that could otherwise couple a future
/// proof to H.
pub(in crate::beta_crown::engine::graph) fn install_child_continuation_state(
    domain: &mut crate::beta_crown::domain::MultiObjectiveGraphBabDomain,
    node_bounds: HashMap<String, Arc<BoundedTensor>>,
    beta_state: GraphBetaState,
    alpha_state: Option<GraphDomainAlphaState>,
    provenance: ChildContinuationStateProvenance,
) {
    domain.node_bounds =
        crate::beta_crown::domain::NodeBoundsMap::from_shared_hash_map(node_bounds);
    domain.delta_pre_nodes.clear();
    domain.beta_state = beta_state;
    if let Some(alpha_state) = alpha_state {
        domain.alpha_state = alpha_state;
    }
    apply_child_continuation_state_provenance(domain, provenance);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SelectivePairRefusal {
    NoActiveObjectives,
    ShapeMismatch,
    ActiveIndexOutOfRange,
    NonFiniteBound,
    CandidateNotStrictlyBetter,
    InvertedInterval,
    AggregateLowerAboveEstablishedUpper,
}

impl SelectivePairRefusal {
    pub(super) const fn telemetry_name(self) -> &'static str {
        match self {
            Self::NoActiveObjectives => "no_active_objectives",
            Self::ShapeMismatch => "shape_mismatch",
            Self::ActiveIndexOutOfRange => "active_index_out_of_range",
            Self::NonFiniteBound => "non_finite_bound",
            Self::CandidateNotStrictlyBetter => "candidate_not_strictly_better",
            Self::InvertedInterval => "inverted_interval",
            Self::AggregateLowerAboveEstablishedUpper => "aggregate_lower_above_established_upper",
        }
    }
}

/// One arm's reported endpoints plus its separately carried continuation state.
///
/// The lower endpoint is independently certified. The W upper endpoint is
/// validated for a well-formed evaluation but never receives publication
/// authority; a W win retains H's upper endpoint instead. The retained H upper
/// is likewise not used as proof authority for a split child.
pub(super) struct ChildArmEvaluation<S> {
    pub(super) bounds: Vec<(f32, f32)>,
    pub(super) continuation_state: S,
}

pub(super) enum SelectivePairDecision<S> {
    Established {
        evaluation: ChildArmEvaluation<S>,
        refusal: SelectivePairRefusal,
    },
    ExpandedWarmup {
        published_bounds: Vec<(f32, f32)>,
        continuation_state: S,
    },
}

/// Select published child bounds and an independently tracked continuation state.
///
/// A W win carries W's node enclosure and warm starts only as future
/// continuation state. It publishes
/// `(max(H_lower, W_lower), H_upper)` for every row: H and W lower endpoints
/// are independently certified, while the β-derived W upper is intentionally
/// ignored. The retained H upper is only the established output endpoint, not a
/// split-child proof certificate, and no claim is made that W produced the
/// published interval. A tie, shape fault, non-finite value, inverted arm
/// interval, or a selected lower above H's upper returns H byte-for-byte.
pub(super) fn select_child_bounds_and_continuation_state<S>(
    established: ChildArmEvaluation<S>,
    candidate: ChildArmEvaluation<S>,
    thresholds: &[f32],
    active_indices: &[usize],
) -> SelectivePairDecision<S> {
    if active_indices.is_empty() {
        return SelectivePairDecision::Established {
            evaluation: established,
            refusal: SelectivePairRefusal::NoActiveObjectives,
        };
    }
    if established.bounds.len() != candidate.bounds.len()
        || established.bounds.len() != thresholds.len()
    {
        return SelectivePairDecision::Established {
            evaluation: established,
            refusal: SelectivePairRefusal::ShapeMismatch,
        };
    }

    for (&(h_lower, h_upper), &(w_lower, w_upper)) in
        established.bounds.iter().zip(&candidate.bounds)
    {
        if !h_lower.is_finite()
            || !h_upper.is_finite()
            || !w_lower.is_finite()
            || !w_upper.is_finite()
        {
            return SelectivePairDecision::Established {
                evaluation: established,
                refusal: SelectivePairRefusal::NonFiniteBound,
            };
        }
        if h_lower > h_upper || w_lower > w_upper {
            return SelectivePairDecision::Established {
                evaluation: established,
                refusal: SelectivePairRefusal::InvertedInterval,
            };
        }
    }

    // Widen before subtraction so two finite f32 endpoints cannot create an
    // infinite comparison score through f32 overflow.
    let mut established_worst = f64::INFINITY;
    let mut candidate_worst = f64::INFINITY;
    for &idx in active_indices {
        let Some(&(h_lower, _)) = established.bounds.get(idx) else {
            return SelectivePairDecision::Established {
                evaluation: established,
                refusal: SelectivePairRefusal::ActiveIndexOutOfRange,
            };
        };
        let Some(&(w_lower, _)) = candidate.bounds.get(idx) else {
            return SelectivePairDecision::Established {
                evaluation: established,
                refusal: SelectivePairRefusal::ActiveIndexOutOfRange,
            };
        };
        let Some(&threshold) = thresholds.get(idx) else {
            return SelectivePairDecision::Established {
                evaluation: established,
                refusal: SelectivePairRefusal::ActiveIndexOutOfRange,
            };
        };
        if !threshold.is_finite() {
            return SelectivePairDecision::Established {
                evaluation: established,
                refusal: SelectivePairRefusal::NonFiniteBound,
            };
        }
        established_worst = established_worst.min(f64::from(h_lower) - f64::from(threshold));
        candidate_worst = candidate_worst.min(f64::from(w_lower) - f64::from(threshold));
    }
    if candidate_worst <= established_worst {
        return SelectivePairDecision::Established {
            evaluation: established,
            refusal: SelectivePairRefusal::CandidateNotStrictlyBetter,
        };
    }

    let mut published_bounds = Vec::with_capacity(established.bounds.len());
    for (&(h_lower, h_upper), &(w_lower, _w_upper)) in
        established.bounds.iter().zip(&candidate.bounds)
    {
        let lower = h_lower.max(w_lower);
        if lower > h_upper {
            return SelectivePairDecision::Established {
                evaluation: established,
                refusal: SelectivePairRefusal::AggregateLowerAboveEstablishedUpper,
            };
        }
        published_bounds.push((lower, h_upper));
    }

    SelectivePairDecision::ExpandedWarmup {
        published_bounds,
        continuation_state: candidate.continuation_state,
    }
}

#[inline]
pub(super) fn selective_candidate_start_allowed(now: Instant, deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|deadline| now < deadline)
}

/// Private W cutoff that preserves the same five-second authoritative GPU tail
/// ordinary chunk admission reserves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SelectiveCandidateDeadlineUnrepresentable;

pub(super) fn selective_candidate_deadline(
    authority_deadline: Option<Instant>,
) -> Result<Option<Instant>, SelectiveCandidateDeadlineUnrepresentable> {
    match authority_deadline {
        None => Ok(None),
        Some(deadline) => deadline
            .checked_sub(SELECTIVE_ROOT_ALPHA_AUTHORITY_RESERVE)
            .map(Some)
            .ok_or(SelectiveCandidateDeadlineUnrepresentable),
    }
}

#[inline]
pub(super) fn authoritative_deadline_expired(
    now: Instant,
    authority_deadline: Option<Instant>,
) -> bool {
    authority_deadline.is_some_and(|deadline| now >= deadline)
}

/// Require a real, sound GPU-resident backend before arming a second shared
/// proof-forest evaluation.
///
/// `engine.is_some()` alone is only a routing hint: CPU/mock engines and
/// unsound WGPU backends also implement `GemmEngine`. Deadline-scored W must
/// additionally use a backend that advertises cooperative bounded dispatch.
pub(super) fn sound_shared_gpu_available(
    engine: &dyn GemmEngine,
    authority_deadline: Option<Instant>,
) -> bool {
    let usable = |gpu: &dyn GpuCrownBackward| {
        gpu.provides_sound_gpu_crown()
            && (authority_deadline.is_none() || gpu.honors_crown_backward_deadline())
    };
    engine.as_gpu_crown_backward().is_some_and(&usable)
        || crate::sound_gpu_gate::sound_gpu_crown_for_wide_with_deadline(authority_deadline)
            .is_some_and(usable)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::beta_crown::branching::GraphNeuronConstraint;
    use crate::{GraphNetwork, GraphNode, Layer, ReLULayer};

    #[test]
    fn gate_is_exactly_one_and_default_off() {
        assert_eq!(
            SelectiveRootAlphaGate::from_raw(None),
            SelectiveRootAlphaGate::Disabled
        );
        assert_eq!(
            SelectiveRootAlphaGate::from_raw(Some("1")),
            SelectiveRootAlphaGate::Enabled
        );
        for raw in ["", "0", "true", "01", "1 ", " 1", "2"] {
            assert_eq!(
                SelectiveRootAlphaGate::from_raw(Some(raw)),
                SelectiveRootAlphaGate::Disabled,
                "{raw:?} must fail closed"
            );
        }
    }

    #[test]
    fn strict_winner_carries_w_continuation_but_keeps_established_upper() {
        let established = ChildArmEvaluation {
            bounds: vec![(-0.5, 1.0), (0.2, 0.9)],
            continuation_state: "H",
        };
        let candidate = ChildArmEvaluation {
            // W's upper endpoints are spuriously much tighter. They are
            // ordered/finite, but split-domain β does not certify them.
            bounds: vec![(-0.3, -0.25), (0.1, 0.11)],
            continuation_state: "W",
        };
        match select_child_bounds_and_continuation_state(established, candidate, &[0.0, 0.0], &[0])
        {
            SelectivePairDecision::ExpandedWarmup {
                published_bounds,
                continuation_state,
            } => {
                assert_eq!(continuation_state, "W", "W continuation must be selected");
                assert_eq!(
                    published_bounds,
                    vec![(-0.3, 1.0), (0.2, 0.9)],
                    "every published upper must remain H byte-for-byte"
                );
            }
            SelectivePairDecision::Established { .. } => panic!("strict W win was rejected"),
        }
    }

    #[test]
    fn regression_returns_established_pair_exactly() {
        let h_bounds = vec![(-0.2, 0.8), (0.3, 0.7)];
        let established = ChildArmEvaluation {
            bounds: h_bounds.clone(),
            continuation_state: "H",
        };
        let candidate = ChildArmEvaluation {
            bounds: vec![(-0.4, 0.6), (0.4, 0.5)],
            continuation_state: "W",
        };
        match select_child_bounds_and_continuation_state(established, candidate, &[0.0, 0.0], &[0])
        {
            SelectivePairDecision::Established {
                evaluation,
                refusal,
            } => {
                assert_eq!(refusal, SelectivePairRefusal::CandidateNotStrictlyBetter);
                assert_eq!(evaluation.continuation_state, "H");
                assert_eq!(evaluation.bounds, h_bounds);
            }
            SelectivePairDecision::ExpandedWarmup { .. } => panic!("regressing W arm won"),
        }
    }

    #[test]
    fn aggregate_lower_above_h_upper_falls_back_exactly() {
        let h_bounds = vec![(-0.2, 0.1)];
        let decision = select_child_bounds_and_continuation_state(
            ChildArmEvaluation {
                bounds: h_bounds.clone(),
                continuation_state: "H",
            },
            ChildArmEvaluation {
                bounds: vec![(0.2, 0.3)],
                continuation_state: "W",
            },
            &[0.0],
            &[0],
        );
        match decision {
            SelectivePairDecision::Established {
                evaluation,
                refusal,
            } => {
                assert_eq!(
                    refusal,
                    SelectivePairRefusal::AggregateLowerAboveEstablishedUpper
                );
                assert_eq!(evaluation.bounds, h_bounds);
                assert_eq!(evaluation.continuation_state, "H");
            }
            SelectivePairDecision::ExpandedWarmup { .. } => {
                panic!("inverted published bounds must fail closed")
            }
        }
    }

    #[test]
    fn malformed_nonfinite_or_inverted_w_returns_h_bits_and_state() {
        let h_bounds = vec![(-0.5_f32, 1.0_f32)];
        for (w_bounds, expected) in [
            (vec![(f32::NAN, 0.5)], SelectivePairRefusal::NonFiniteBound),
            (
                vec![(0.0, f32::INFINITY)],
                SelectivePairRefusal::NonFiniteBound,
            ),
            (vec![(0.4, 0.3)], SelectivePairRefusal::InvertedInterval),
        ] {
            let decision = select_child_bounds_and_continuation_state(
                ChildArmEvaluation {
                    bounds: h_bounds.clone(),
                    continuation_state: 0x4848_u16,
                },
                ChildArmEvaluation {
                    bounds: w_bounds,
                    continuation_state: 0x5757_u16,
                },
                &[0.0],
                &[0],
            );
            let SelectivePairDecision::Established {
                evaluation,
                refusal,
            } = decision
            else {
                panic!("malformed W acquired authority");
            };
            assert_eq!(refusal, expected);
            assert_eq!(evaluation.continuation_state, 0x4848);
            assert_eq!(
                evaluation
                    .bounds
                    .iter()
                    .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
                    .collect::<Vec<_>>(),
                h_bounds
                    .iter()
                    .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn only_selected_warmup_invalidates_state_dependent_caches() {
        assert!(!ChildContinuationStateProvenance::Established.invalidates_all_cached_las());
        assert!(
            ChildContinuationStateProvenance::SelectiveExpandedWarmup.invalidates_all_cached_las()
        );

        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid input");
        let mut domain = crate::beta_crown::domain::MultiObjectiveGraphBabDomain::root(
            HashMap::new(),
            vec![(-1.0, 1.0), (-0.5, 0.5)],
            &input,
            &[0.0, 0.0],
            false,
        )
        .expect("valid domain");
        domain.cached_las[0] = Some(Arc::new(
            crate::batched_domain::CachedLinearBounds::default(),
        ));
        domain.set_per_disjunct_alphas(vec![
            GraphDomainAlphaState::empty(),
            GraphDomainAlphaState::empty(),
        ]);

        apply_child_continuation_state_provenance(
            &mut domain,
            ChildContinuationStateProvenance::Established,
        );
        assert!(domain.cached_las()[0].is_some());
        assert!(domain.per_disjunct_alphas().is_some());

        apply_child_continuation_state_provenance(
            &mut domain,
            ChildContinuationStateProvenance::SelectiveExpandedWarmup,
        );
        assert!(domain.cached_las().iter().all(Option::is_none));
        assert!(domain.per_disjunct_alphas().is_none());
    }

    #[test]
    fn w_win_installs_continuation_and_next_child_inherits_only_w_state() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.set_output("relu");
        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32, -1.0, -1.0]).into_dyn(),
            ndarray::arr1(&[1.0_f32, 1.0, 1.0]).into_dyn(),
        )
        .expect("valid input");
        let root_nodes = graph.collect_node_bounds(&input).expect("root bounds");
        let root = crate::beta_crown::domain::MultiObjectiveGraphBabDomain::root(
            root_nodes,
            vec![(-1.0, 1.0)],
            &input,
            &[0.0],
            false,
        )
        .expect("root domain");
        let mut child = root
            .with_constraint(
                &graph,
                GraphNeuronConstraint {
                    node_name: "relu".to_string(),
                    neuron_idx: 0,
                    is_active: true,
                    score: 1.0,
                },
                false,
                &[0.0],
            )
            .expect("first split")
            .expect("feasible first child");
        child.cached_las[0] = Some(Arc::new(
            crate::batched_domain::CachedLinearBounds::default(),
        ));
        child.set_per_disjunct_alphas(vec![GraphDomainAlphaState::empty()]);

        let mut w_beta = GraphBetaState::from_history(child.history()).expect("W beta");
        w_beta.entries[0].set_value(0.73);
        let mut w_alpha = child.alpha_state().clone();
        w_alpha
            .neurons
            .get_mut("relu")
            .and_then(|neurons| neurons.get_mut(&2))
            .expect("third neuron remains unstable")
            .set_alpha(0.37);
        w_alpha
            .upper_neurons
            .get_mut("relu")
            .and_then(|neurons| neurons.get_mut(&2))
            .expect("third upper neuron remains unstable")
            .set_alpha(0.63);
        let w_nodes: HashMap<_, _> = graph
            .collect_node_bounds(&input)
            .expect("independent W node cache")
            .into_iter()
            .map(|(name, bounds)| (name, Arc::new(bounds)))
            .collect();
        let w_relu_arc = Arc::clone(w_nodes.get("relu").expect("W relu cache"));

        let decision = select_child_bounds_and_continuation_state(
            ChildArmEvaluation {
                bounds: vec![(-0.5, 1.0)],
                continuation_state: (
                    child.node_bounds.to_shared_hash_map(),
                    child.beta_state().clone(),
                    Some(child.alpha_state().clone()),
                    ChildContinuationStateProvenance::Established,
                ),
            },
            ChildArmEvaluation {
                // W upper is deliberately tighter but not authoritative.
                bounds: vec![(-0.1, 0.2)],
                continuation_state: (
                    w_nodes,
                    w_beta,
                    Some(w_alpha),
                    ChildContinuationStateProvenance::SelectiveExpandedWarmup,
                ),
            },
            &[0.0],
            &[0],
        );
        let SelectivePairDecision::ExpandedWarmup {
            published_bounds,
            continuation_state: (w_nodes, w_beta, w_alpha, provenance),
        } = decision
        else {
            panic!("strict W lower-margin win must select W continuation");
        };
        assert_eq!(published_bounds, vec![(-0.1, 1.0)]);
        install_child_continuation_state(&mut child, w_nodes, w_beta, w_alpha, provenance);
        child
            .update_bounds(published_bounds, &[0.0], false)
            .expect("published bounds install separately");

        assert!(Arc::ptr_eq(
            child.node_bounds().get("relu").expect("installed W cache"),
            &w_relu_arc
        ));
        assert!((child.beta_state().entries[0].value() - 0.73).abs() < f32::EPSILON);
        assert!((child.alpha_state().alpha("relu", 2) - 0.37).abs() < f32::EPSILON);
        assert!(child.cached_las().iter().all(Option::is_none));
        assert!(child.per_disjunct_alphas().is_none());

        let next = child
            .with_constraint(
                &graph,
                GraphNeuronConstraint {
                    node_name: "relu".to_string(),
                    neuron_idx: 1,
                    is_active: false,
                    score: 1.0,
                },
                false,
                &[0.0],
            )
            .expect("second split")
            .expect("feasible grandchild");
        assert!(Arc::ptr_eq(
            next.node_bounds().get("relu").expect("inherited W cache"),
            &w_relu_arc
        ));
        assert!(
            (next.beta_state().entries[0].value() - 0.73).abs() < f32::EPSILON,
            "next child must warm-start the existing split from W beta"
        );
        assert!(
            (next.alpha_state().alpha("relu", 2) - 0.37).abs() < f32::EPSILON,
            "next child must inherit W alpha for a still-unstable neuron"
        );
        assert!(next.cached_las().iter().all(Option::is_none));
        assert!(next.per_disjunct_alphas().is_none());
    }

    #[test]
    fn optional_candidate_preserves_five_second_gpu_tail() {
        let now = Instant::now();
        assert!(selective_candidate_start_allowed(now, None));
        let authority = now + Duration::from_secs(7);
        let private = selective_candidate_deadline(Some(authority))
            .expect("reserve can be subtracted")
            .expect("scored call has a private deadline");
        assert_eq!(private, now + Duration::from_secs(2));
        assert!(selective_candidate_start_allowed(now, Some(private)));
        assert!(!selective_candidate_start_allowed(private, Some(private)));
        assert!(!selective_candidate_start_allowed(
            now,
            selective_candidate_deadline(Some(now + Duration::from_secs(5)))
                .expect("exact reserve subtracts")
        ));
        assert!(
            selective_candidate_deadline(Some(now + Duration::from_secs(4))).is_ok(),
            "Instant arithmetic itself remains representable"
        );
        assert!(authoritative_deadline_expired(authority, Some(authority)));
        assert!(!authoritative_deadline_expired(private, Some(authority)));
    }
}
