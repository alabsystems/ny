// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DAG gradient dispatch delegation for the root alpha collection path.
//!
//! Part of #4036: when `config.gradient_method` is not SPSA, the root
//! collection loop delegates to the full DAG optimizer which properly
//! dispatches AnalyticChain, FD, and Analytic gradient methods.

use super::*;
use crate::bounds::GradientMethod;

/// Shrink-only intersection for the typed cGAN publication boundary.
///
/// `BoundedTensor::intersection_per_element` deliberately returns a sound
/// union on disjoint elements. That diagnostic policy is appropriate for the
/// generic route below, but the typed transaction promises never to widen its
/// certified forward-linear baseline. Reject the whole optimized-output
/// candidate on any disjoint element or malformed tensor.
fn typed_cgan_output_intersection(
    baseline: &BoundedTensor,
    optimized: &BoundedTensor,
) -> Option<BoundedTensor> {
    baseline
        .intersection_per_element(optimized)
        .and_then(|(tightened, disjoint)| (disjoint == 0).then_some(tightened))
}

impl GraphNetwork {
    /// Try to delegate alpha collection to the DAG optimizer for non-SPSA
    /// gradient methods.
    ///
    /// Returns `Ok(Some(result))` if delegation succeeded, `Ok(None)` if SPSA
    /// is configured (the caller should fall through to the SPSA loop).
    pub(in crate::network::graph_alpha) fn try_dag_gradient_dispatch(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn ny_core::GemmEngine>,
        exec_order: &[String],
        precomputed_reference: Option<PrecomputedAlphaReferenceBounds>,
    ) -> Result<Option<GraphAlphaCollectionResult>> {
        self.try_dag_gradient_dispatch_with_phase_cap_policy(
            input,
            config,
            engine,
            exec_order,
            precomputed_reference,
            false,
        )
        .map(|outcome| outcome.map(GraphAlphaCollectionOutcome::into_result))
    }

    /// Root-only variant that may publish a fully returned DAG-alpha artifact
    /// when its local phase deadline has expired.  The caller separately owns
    /// the global verifier deadline and decides whether that checkpoint still
    /// has time to be re-evaluated.
    pub(in crate::network::graph_alpha) fn try_dag_gradient_dispatch_with_phase_cap_checkpoint(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn ny_core::GemmEngine>,
        exec_order: &[String],
    ) -> Result<Option<GraphAlphaCollectionOutcome>> {
        self.try_dag_gradient_dispatch_with_phase_cap_policy(
            input, config, engine, exec_order, None, true,
        )
    }

    fn try_dag_gradient_dispatch_with_phase_cap_policy(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn ny_core::GemmEngine>,
        exec_order: &[String],
        precomputed_reference: Option<PrecomputedAlphaReferenceBounds>,
        allow_phase_cap_checkpoint: bool,
    ) -> Result<Option<GraphAlphaCollectionOutcome>> {
        if config.gradient_method == GradientMethod::Spsa {
            return Ok(None);
        }

        let collected = match precomputed_reference {
            Some(reference) => self.propagate_dag_alpha_crown_collect_with_engine_and_reference(
                input, config, engine, reference,
            )?,
            None if phase_cap_collection_retention_enabled(
                allow_phase_cap_checkpoint,
                config.fix_interm_bounds,
            ) =>
            {
                self.propagate_dag_alpha_crown_collect_with_engine_phase_cap_checkpoint(
                    input, config, engine,
                )?
            }
            None => self.propagate_dag_alpha_crown_collect_with_engine(input, config, engine)?,
        };
        let artifact = match collected {
            Some(artifact) => artifact,
            None => return Ok(None),
        };
        let super::super::propagate_dag::DagAlphaCollectionArtifact {
            output_bounds,
            alpha_state,
            reference_bounds: artifact_reference_bounds,
            reference_bounds_source,
            completed_iterations,
            optimizer_updates_completed,
        } = artifact;
        let cgan_complete_crown_ibp = matches!(
            reference_bounds_source,
            AlphaReferenceBoundsSource::CganCompleteCrownIbp { .. }
        );
        let cgan_sparse_target_complete = matches!(
            reference_bounds_source,
            AlphaReferenceBoundsSource::CganSparseTargetComplete { .. }
        );
        let checkpoint_candidate = phase_cap_checkpoint_candidate(
            allow_phase_cap_checkpoint,
            config.fix_interm_bounds,
            completed_iterations,
        );

        // The typed cGAN collectors are intentionally absent from the generic
        // collection cache because both start from a caller-supplied baseline.
        // Reuse the optimizer's final sound reference map for either exact
        // route; otherwise this unconditional recollection starts a second
        // identical root transaction before the returned alpha state is
        // evaluated. On the complete route that second pass was observed with
        // only 54 seconds left after the first decisive 101-second cascade.
        //
        // Keep every ordinary route on its historical recollection contract.
        let typed_reference = cgan_sparse_target_complete || cgan_complete_crown_ibp;
        let phase_deadline_expired = checkpoint_candidate && config.past_deadline();
        let (reference_bounds, phase_cap_checkpoint) = match resolve_reference_bounds_publication(
            artifact_reference_bounds,
            typed_reference,
            checkpoint_candidate,
            phase_deadline_expired,
            || self.collect_alpha_reference_bounds_with_engine(input, config, engine, exec_order),
        )? {
            ReferenceBoundsPublication::Complete(bounds) => (bounds, false),
            ReferenceBoundsPublication::PhaseCapCheckpoint(bounds) => (bounds, true),
        };
        if cgan_sparse_target_complete {
            // This route's authority is deliberately one atomic target over a
            // complete forward-linear baseline. Running the ordinary
            // `fix_interm_bounds=false` post-loop collection here widens that
            // authority to every intermediate and, on the official cGAN row,
            // spent the remaining 167-second root window walking 28,800-row
            // targets after the selected 428-row transaction had completed.
            //
            // Publish the optimizer-owned sound map directly, intersecting its
            // output image with the optimized output bound when shapes agree.
            // Every retained intermediate remains no wider than the certified
            // baseline, the optimized output is still used, and no second
            // intermediate transaction can consume the proof phase.
            let mut merged = reference_bounds;
            let output_node = if self.output_node.is_empty() {
                exec_order.last().map(String::as_str).unwrap_or("")
            } else {
                &self.output_node
            };
            if let Some(existing) = merged.get(output_node) {
                if existing.shape() == output_bounds.shape() {
                    if let Some(tightened) =
                        typed_cgan_output_intersection(existing, &output_bounds)
                    {
                        merged.insert(output_node.to_string(), tightened);
                    }
                }
            }
            let result = (merged, alpha_state);
            return Ok(Some(collection_outcome(
                result,
                phase_cap_checkpoint,
                completed_iterations,
                optimizer_updates_completed,
            )));
        }
        if config.fix_interm_bounds {
            let result = (reference_bounds, alpha_state);
            return Ok(Some(collection_outcome(
                result,
                phase_cap_checkpoint,
                completed_iterations,
                optimizer_updates_completed,
            )));
        }

        // Collect bounds at all nodes using the optimized alpha state.
        let all_bounds = self.collect_crown_bounds_with_alpha(
            input,
            &reference_bounds,
            &alpha_state,
            engine,
            config.deadline,
        )?;

        // Intersect the DAG optimizer's output bounds with the collection bounds.
        let output_node = if self.output_node.is_empty() {
            exec_order.last().map(String::as_str).unwrap_or("")
        } else {
            &self.output_node
        };
        let mut merged = all_bounds;
        if let Some(existing) = merged.get(output_node) {
            if existing.shape() == output_bounds.shape() {
                if let Some((tightened, _)) = existing.intersection_per_element(&output_bounds) {
                    merged.insert(output_node.to_string(), tightened);
                }
            }
        }

        Ok(Some(GraphAlphaCollectionOutcome::Complete((
            merged,
            alpha_state,
        ))))
    }
}

fn phase_cap_checkpoint_candidate(
    policy_enabled: bool,
    fix_interm_bounds: bool,
    completed_iterations: usize,
) -> bool {
    phase_cap_collection_retention_enabled(policy_enabled, fix_interm_bounds)
        && completed_iterations > 0
}

fn phase_cap_collection_retention_enabled(policy_enabled: bool, fix_interm_bounds: bool) -> bool {
    policy_enabled && fix_interm_bounds
}

enum ReferenceBoundsPublication {
    Complete(std::collections::HashMap<String, BoundedTensor>),
    PhaseCapCheckpoint(std::collections::HashMap<String, BoundedTensor>),
}

fn resolve_reference_bounds_publication<F>(
    artifact_reference_bounds: std::collections::HashMap<String, BoundedTensor>,
    typed_reference: bool,
    checkpoint_candidate: bool,
    phase_deadline_expired: bool,
    recollect: F,
) -> Result<ReferenceBoundsPublication>
where
    F: FnOnce() -> Result<std::collections::HashMap<String, BoundedTensor>>,
{
    if typed_reference {
        return Ok(ReferenceBoundsPublication::Complete(
            artifact_reference_bounds,
        ));
    }
    if checkpoint_candidate && phase_deadline_expired {
        return Ok(ReferenceBoundsPublication::PhaseCapCheckpoint(
            artifact_reference_bounds,
        ));
    }
    match recollect() {
        Ok(bounds) => Ok(ReferenceBoundsPublication::Complete(bounds)),
        Err(NyError::DeadlineExceeded(_)) if checkpoint_candidate => Ok(
            ReferenceBoundsPublication::PhaseCapCheckpoint(artifact_reference_bounds),
        ),
        Err(error) => Err(error),
    }
}

fn collection_outcome(
    result: GraphAlphaCollectionResult,
    phase_cap_checkpoint: bool,
    completed_iterations: usize,
    optimizer_updates_completed: usize,
) -> GraphAlphaCollectionOutcome {
    if phase_cap_checkpoint {
        GraphAlphaCollectionOutcome::PhaseCapCheckpoint {
            result,
            completed_iterations,
            optimizer_updates_completed,
        }
    } else {
        GraphAlphaCollectionOutcome::Complete(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        phase_cap_checkpoint_candidate, phase_cap_collection_retention_enabled,
        resolve_reference_bounds_publication, typed_cgan_output_intersection,
        ReferenceBoundsPublication,
    };
    use crate::bounds::{AlphaCrownConfig, GradientMethod};
    use crate::layers::{AddLayer, Layer, LinearLayer, ReLULayer};
    use crate::network::{GraphNetwork, GraphNode};
    use ndarray::{arr1, arr2};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;
    use std::collections::{BTreeMap, HashMap};

    fn bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
            .expect("valid test bounds")
    }

    fn checkpoint_fixture() -> (GraphNetwork, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "left_linear",
            Layer::Linear(
                LinearLayer::new(
                    arr2(&[[1.0_f32, -0.4], [0.7, 0.9]]),
                    Some(arr1(&[0.1_f32, -0.2])),
                )
                .expect("left linear"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "left_relu",
            Layer::ReLU(ReLULayer),
            vec!["left_linear".to_string()],
        ));
        graph.add_node(GraphNode::from_input(
            "right_linear",
            Layer::Linear(
                LinearLayer::new(
                    arr2(&[[-0.6_f32, 1.1], [0.8, -0.5]]),
                    Some(arr1(&[-0.15_f32, 0.05])),
                )
                .expect("right linear"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "right_relu",
            Layer::ReLU(ReLULayer),
            vec!["right_linear".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "residual",
            Layer::Add(AddLayer),
            "left_relu",
            "right_relu",
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.2_f32, -0.9]]), Some(arr1(&[0.07_f32])))
                    .expect("output linear"),
            ),
            vec!["residual".to_string()],
        ));
        graph.set_output("output");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.7]).into_dyn(),
            arr1(&[1.3_f32, 1.1]).into_dyn(),
        )
        .expect("input box");
        (graph, input)
    }

    fn bound_map_bits(
        map: &HashMap<String, BoundedTensor>,
    ) -> BTreeMap<String, (Vec<u32>, Vec<u32>)> {
        map.iter()
            .map(|(name, bound)| {
                (
                    name.clone(),
                    (
                        bound.lower().iter().map(|v| v.to_bits()).collect(),
                        bound.upper().iter().map(|v| v.to_bits()).collect(),
                    ),
                )
            })
            .collect()
    }

    #[test]
    fn typed_cgan_output_publication_is_strictly_shrink_only() {
        let baseline = bounds(&[0.0, -2.0], &[2.0, 4.0]);
        let overlapping = bounds(&[1.0, -3.0], &[3.0, 1.0]);
        let tightened = typed_cgan_output_intersection(&baseline, &overlapping)
            .expect("fully overlapping candidate");
        assert_eq!(tightened.lower(), &arr1(&[1.0, -2.0]).into_dyn());
        assert_eq!(tightened.upper(), &arr1(&[2.0, 1.0]).into_dyn());

        let partly_disjoint = bounds(&[1.0, 5.0], &[3.0, 6.0]);
        assert!(
            typed_cgan_output_intersection(&baseline, &partly_disjoint).is_none(),
            "a per-element union fallback would widen the certified baseline"
        );
    }

    #[test]
    fn phase_cap_checkpoint_requires_every_publication_predicate() {
        assert!(phase_cap_checkpoint_candidate(true, true, 1));
        for candidate in [(false, true, 1), (true, false, 1), (true, true, 0)] {
            assert!(
                !phase_cap_checkpoint_candidate(candidate.0, candidate.1, candidate.2),
                "incomplete checkpoint predicate unexpectedly admitted {candidate:?}"
            );
        }
    }

    #[test]
    fn fixed_intermediates_are_required_before_deadline_retention_enters_optimizer() {
        assert!(phase_cap_collection_retention_enabled(true, true));
        assert!(!phase_cap_collection_retention_enabled(false, true));
        assert!(!phase_cap_collection_retention_enabled(true, false));
        assert!(!phase_cap_collection_retention_enabled(false, false));
    }

    #[test]
    fn late_tail_recollection_retains_only_the_completed_artifact_map() {
        let artifact_bound = bounds(&[-2.0], &[3.0]);
        let mut artifact = HashMap::new();
        artifact.insert("output".to_string(), artifact_bound.clone());
        let publication =
            resolve_reference_bounds_publication(artifact, false, true, false, || {
                Err(NyError::DeadlineExceeded("synthetic late tail".into()))
            })
            .expect("a completed artifact survives a late tail recollection");
        let ReferenceBoundsPublication::PhaseCapCheckpoint(retained) = publication else {
            panic!("late tail must produce a typed phase-cap checkpoint");
        };
        let retained = retained.get("output").expect("artifact output retained");
        assert_eq!(retained.lower(), artifact_bound.lower());
        assert_eq!(retained.upper(), artifact_bound.upper());
    }

    #[test]
    fn expired_cap_skips_tail_but_disabled_policy_preserves_deadline_error() {
        let mut artifact = HashMap::new();
        artifact.insert("output".to_string(), bounds(&[-2.0], &[3.0]));
        let publication =
            resolve_reference_bounds_publication(artifact.clone(), false, true, true, || {
                panic!("an already-expired phase cap must not launch the tail")
            })
            .expect("eligible expired cap should retain the artifact");
        assert!(matches!(
            publication,
            ReferenceBoundsPublication::PhaseCapCheckpoint(_)
        ));

        let error = match resolve_reference_bounds_publication(artifact, false, false, true, || {
            Err(NyError::DeadlineExceeded("legacy tail deadline".into()))
        }) {
            Ok(_) => panic!("gate-off behavior must preserve the legacy deadline error"),
            Err(error) => error,
        };
        assert!(error.is_deadline_exceeded());
    }

    #[test]
    fn expiry_before_first_completed_fold_preserves_original_deadline() {
        let artifact = HashMap::from([("output".to_string(), bounds(&[-2.0], &[3.0]))]);
        let checkpoint_candidate = phase_cap_checkpoint_candidate(true, true, 0);
        let error = match resolve_reference_bounds_publication(
            artifact,
            false,
            checkpoint_candidate,
            true,
            || Err(NyError::DeadlineExceeded("before first finite fold".into())),
        ) {
            Ok(_) => panic!("a zero-fold artifact must never be published"),
            Err(error) => error,
        };
        match error {
            NyError::DeadlineExceeded(message) => {
                assert_eq!(message, "before first finite fold")
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn typed_reference_remains_complete_even_when_phase_cap_is_expired() {
        let expected = bounds(&[-2.0], &[3.0]);
        let artifact = HashMap::from([("output".to_string(), expected.clone())]);
        let publication = resolve_reference_bounds_publication(artifact, true, true, true, || {
            panic!("a typed reference must never launch ordinary recollection")
        })
        .expect("typed artifact publication");
        let ReferenceBoundsPublication::Complete(bounds) = publication else {
            panic!("typed reference must not be labeled a phase checkpoint")
        };
        assert_eq!(bounds["output"].lower(), expected.lower());
        assert_eq!(bounds["output"].upper(), expected.upper());
    }

    #[ntest::timeout(30000)]
    #[test]
    fn retained_real_dag_artifact_re_evaluates_soundly_without_mutating_payload() {
        let (graph, input) = checkpoint_fixture();
        let config = AlphaCrownConfig {
            iterations: 1,
            gradient_method: GradientMethod::AnalyticChain,
            fix_interm_bounds: true,
            adaptive_skip: false,
            adaptive_skip_pilot: false,
            ..AlphaCrownConfig::default()
        };
        let artifact = graph
            .propagate_dag_alpha_crown_collect_with_engine_phase_cap_checkpoint(
                &input, &config, None,
            )
            .expect("real DAG collection")
            .expect("fixture has optimizable ReLUs");
        assert_eq!(artifact.completed_iterations, 1);
        assert_eq!(artifact.optimizer_updates_completed, 1);

        let expected_map_bits = bound_map_bits(&artifact.reference_bounds);
        let alpha = artifact.alpha_state;
        let alpha_bits_before = alpha
            .alphas
            .iter()
            .map(|(name, values)| {
                (
                    name.clone(),
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let publication = resolve_reference_bounds_publication(
            artifact.reference_bounds,
            false,
            true,
            false,
            || Err(NyError::DeadlineExceeded("synthetic tail expiry".into())),
        )
        .expect("completed real artifact should be retained");
        let ReferenceBoundsPublication::PhaseCapCheckpoint(reference_bounds) = publication else {
            panic!("synthetic late tail must publish the real checkpoint")
        };
        assert_eq!(bound_map_bits(&reference_bounds), expected_map_bits);

        let reevaluated = graph
            .collect_crown_bounds_with_alpha(&input, &reference_bounds, &alpha, None, None)
            .expect("retained map/state must support fresh certified evaluation");
        let output = &reevaluated["output"];
        let lower = *output.lower().iter().next().expect("scalar lower");
        let upper = *output.upper().iter().next().expect("scalar upper");
        assert!(lower.is_finite() && upper.is_finite() && lower <= upper);

        for x0 in [-1.0_f32, 0.15, 1.3] {
            for x1 in [-0.7_f32, 0.2, 1.1] {
                let left = [
                    (x0 - 0.4 * x1 + 0.1).max(0.0),
                    (0.7 * x0 + 0.9 * x1 - 0.2).max(0.0),
                ];
                let right = [
                    (-0.6 * x0 + 1.1 * x1 - 0.15).max(0.0),
                    (0.8 * x0 - 0.5 * x1 + 0.05).max(0.0),
                ];
                let exact = 1.2 * (left[0] + right[0]) - 0.9 * (left[1] + right[1]) + 0.07;
                assert!(
                    lower <= exact && exact <= upper,
                    "sample ({x0}, {x1})={exact} escaped [{lower}, {upper}]"
                );
            }
        }
        let alpha_bits_after = alpha
            .alphas
            .iter()
            .map(|(name, values)| {
                (
                    name.clone(),
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(alpha_bits_after, alpha_bits_before);
    }
}
