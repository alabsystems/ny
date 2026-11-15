// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Paid scheduling for the original raw root. These times grant no proof
//! authority: all bounds still come from the existing certified evaluators.

use crate::beta_crown::config::VerificationArtifactAuthority;
use crate::bounds::GradientMethod;
use crate::{BetaCrownConfig, GraphNetwork};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub(super) struct Plan {
    pub bootstrap: Instant,
    pub tighten: Instant,
    pub spec: Instant,
    pub reference_cap: Duration,
}

fn split(now: Instant, end: Instant, reference_fraction: f32) -> Option<Plan> {
    if !reference_fraction.is_finite() || reference_fraction <= 0.0 {
        return None;
    }
    let remaining = end.checked_duration_since(now)?;
    let half = remaining / 2;
    let reference_cap = remaining.mul_f32(reference_fraction.min(0.25));
    if half.is_zero() || reference_cap.is_zero() {
        return None;
    }
    Some(Plan {
        bootstrap: now + half,
        tighten: end.checked_sub((remaining / 8) * 3)?,
        spec: end.checked_sub(remaining / 8)?,
        reference_cap,
    })
}

pub(super) fn plan(
    config: &BetaCrownConfig,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    now: Instant,
    authority: Instant,
) -> Option<Plan> {
    if !config.certified_root_cut_raw_source
        || config.verification_artifact_authority != VerificationArtifactAuthority::VerdictOnly
        || config.crown_backward_layers.is_some()
        || !config.use_alpha_crown
        || !config.alpha_config.fix_interm_bounds
        || config.alpha_config.gradient_method == GradientMethod::Spsa
        || config.alpha_config.invprop.enabled
        || config.alpha_config.iterations == 0
        // This branch has a separate large-convolution collection policy.
        || (graph.has_conv_layers() && input.len() > 5000)
    {
        return None;
    }
    let owner_end = config
        .certified_root_cut_owner
        .as_ref()?
        .bootstrap_deadline()?;
    split(
        now,
        authority.min(owner_end),
        config.phase_budget.initial_bounds_fraction,
    )
}

/// A fresh objective pass gets half the live reserved root window. Its expired
/// bootstrap clock is irrelevant; the other half remains available to cuts and
/// the rest of root evaluation. Never manufacture time after authority expiry.
pub(super) fn objective_deadline(now: Instant, authority: Instant) -> Instant {
    now.checked_add(authority.saturating_duration_since(now) / 2)
        .unwrap_or(authority)
        .min(authority)
}

pub(super) fn may_reuse_after(error: &NyError) -> bool {
    matches!(error, NyError::DeadlineExceeded(_))
        || super::super::super::is_finite_constrained_crown_refusal(error)
}

/// Projection of a previously certified output needs the live outer clock,
/// never a restarted alpha/IBP transaction or the exhausted local spec clock.
pub(super) fn project(
    graph: &GraphNetwork,
    bootstrap: &super::GraphBabBootstrap,
    objectives: &[Vec<f32>],
    authority: Option<Instant>,
) -> Result<(BoundedTensor, Vec<(f32, f32)>)> {
    let authority = authority.ok_or_else(|| {
        NyError::InvalidSpec(
            "retained root reference requires a finite objective authority".to_string(),
        )
    })?;
    if Instant::now() >= authority {
        return Err(NyError::DeadlineExceeded(
            "retained root reference authority expired".to_string(),
        ));
    }
    let output = super::bootstrap_output_bounds(graph, bootstrap).ok_or_else(|| {
        NyError::InvalidSpec("retained root reference is missing the graph output".to_string())
    })?;
    let bounds =
        crate::beta_crown::engine::graph::objectives::objective_bounds_multi_with_deadline(
            output, objectives, authority,
        )?;
    if Instant::now() >= authority {
        return Err(NyError::DeadlineExceeded(
            "retained root projection authority expired".to_string(),
        ));
    }
    Ok((output.clone(), bounds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_reserves_objective_cut_and_search_under_one_clock() {
        let now = Instant::now();
        let end = now + Duration::from_secs(80);
        let p = split(now, end, 0.15).unwrap();
        assert_eq!(p.bootstrap - now, Duration::from_secs(40));
        assert_eq!(p.tighten - now, Duration::from_secs(50));
        assert_eq!(p.spec - now, Duration::from_secs(70));
        assert!((p.reference_cap.as_secs_f64() - 12.0).abs() < 0.001);
        assert_eq!(
            objective_deadline(p.tighten, p.spec) - now,
            Duration::from_secs(60)
        );
    }

    #[test]
    fn expired_and_invalid_windows_cannot_mint_time() {
        let now = Instant::now();
        assert!(split(now, now, 0.15).is_none());
        assert!(split(now, now + Duration::from_secs(80), f32::NAN).is_none());
        assert!(split(now, now + Duration::from_secs(80), 0.0).is_none());
        assert_eq!(
            objective_deadline(now, now.checked_sub(Duration::from_secs(1)).unwrap()),
            now.checked_sub(Duration::from_secs(1)).unwrap()
        );
        assert_eq!(
            split(now, now + Duration::from_secs(80), 1.0)
                .unwrap()
                .reference_cap,
            Duration::from_secs(20)
        );
    }

    #[test]
    fn retained_fallback_does_not_mask_invalid_bounds_or_unknown_failures() {
        assert!(may_reuse_after(&NyError::DeadlineExceeded("local".into())));
        assert!(!may_reuse_after(&NyError::InvalidSpec("invalid".into())));
        assert!(!may_reuse_after(&NyError::InternalError("invalid".into())));
        assert!(!may_reuse_after(&NyError::UnsupportedConfiguration(
            "unknown".into()
        )));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::beta_crown::certified_root_cut::QueryOwner;
    use crate::beta_crown::engine::graph::shared::init::{
        compute_graph_bab_bootstrap_with_bounded_fixed_reference, GraphBabBootstrap,
    };
    use crate::bounds::GradientMethod;
    use crate::{BetaCrownVerifier, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, arr2};

    // Both authored outputs equal ReLU(x). Independent output-box projection
    // loses that equality; a genuine same-map C fold can recover it exactly.
    fn fixture() -> (GraphNetwork, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "pre",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["pre".into()],
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32], [1.0]]), None).unwrap()),
            vec!["relu".into()],
        ));
        graph.set_output("output");
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        (graph, input)
    }

    fn config(end: Instant) -> BetaCrownConfig {
        let mut cfg = BetaCrownConfig {
            use_alpha_crown: true,
            use_forward_bounds: false,
            verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
            certified_root_cut_raw_source: true,
            certified_root_cut_owner: QueryOwner::new(Some(end), true),
            ..BetaCrownConfig::default()
        };
        cfg.alpha_config.iterations = 1;
        cfg.alpha_config.gradient_method = GradientMethod::AnalyticChain;
        cfg.alpha_config.fix_interm_bounds = true;
        cfg.alpha_config.adaptive_skip = false;
        cfg.alpha_config.adaptive_skip_pilot = false;
        cfg.alpha_config.invprop.enabled = false;
        cfg.phase_budget.initial_bounds_fraction = 0.15;
        cfg
    }

    fn bootstrap(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        cfg: &BetaCrownConfig,
    ) -> GraphBabBootstrap {
        let now = Instant::now();
        compute_graph_bab_bootstrap_with_bounded_fixed_reference(
            graph,
            input,
            cfg,
            None,
            now + Duration::from_secs(30),
            now + Duration::from_secs(60),
            Duration::from_secs(5),
        )
        .expect("genuine bounded reference and completed alpha fold")
    }

    fn bits(output: &BoundedTensor) -> Vec<u32> {
        output
            .lower()
            .iter()
            .chain(output.upper().iter())
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn bounded_plan_rejects_ineligible_scopes_and_clips_to_owner() {
        let (graph, input) = fixture();
        let now = Instant::now();
        let owner_end = now + Duration::from_secs(60);
        let authority = now + Duration::from_secs(80);
        let cfg = config(owner_end);
        let admitted = plan(&cfg, &graph, &input, now, authority).expect("eligible original root");
        assert_eq!(admitted.bootstrap, now + Duration::from_secs(30));
        assert_eq!(
            admitted.spec,
            owner_end.checked_sub(Duration::from_millis(7500)).unwrap()
        );
        assert!((admitted.reference_cap.as_secs_f64() - 9.0).abs() < 0.001);
        for case in 0..11 {
            let mut refused = cfg.clone();
            match case {
                0 => refused.certified_root_cut_raw_source = false,
                1 => {
                    refused.verification_artifact_authority =
                        VerificationArtifactAuthority::CertificateExport
                }
                2 => refused.crown_backward_layers = Some(1),
                3 => refused.use_alpha_crown = false,
                4 => refused.alpha_config.fix_interm_bounds = false,
                5 => refused.alpha_config.gradient_method = GradientMethod::Spsa,
                6 => refused.alpha_config.invprop.enabled = true,
                7 => refused.alpha_config.iterations = 0,
                8 => refused.certified_root_cut_owner = None,
                9 => refused.certified_root_cut_owner = QueryOwner::new(Some(owner_end), false),
                10 => refused.phase_budget.initial_bounds_fraction = f32::NAN,
                _ => unreachable!(),
            }
            assert!(
                plan(&refused, &graph, &input, now, authority).is_none(),
                "guard {case}"
            );
        }
        assert!(plan(&cfg, &graph, &input, owner_end, authority).is_none());
        assert!(plan(&cfg, &graph, &input, now, now).is_none());
    }

    #[ntest::timeout(30000)]
    #[test]
    fn bounded_bootstrap_fresh_objectives_use_one_reference_after_local_expiry() {
        for expired in [false, true] {
            let (graph, input) = fixture();
            let authority = Instant::now() + Duration::from_secs(60);
            let cfg = config(authority);
            let counter = GraphNetwork::root_fixed_reference_collection_counter_for_test();
            let mut retained = bootstrap(&graph, &input, &cfg);
            assert!(retained.retained_fixed_reference);
            assert!(
                retained.phase_cap_optimizer_updates.is_none(),
                "normal completion still owns reusable reference"
            );
            assert_eq!(counter(), 1);
            let objectives = vec![vec![1.0_f32, -1.0], vec![-1.0, 1.0]];
            let expected_output = bits(&retained.initial_node_bounds["output"]);
            let (_, projected) = project(&graph, &retained, &objectives, Some(authority)).unwrap();
            assert!(projected.iter().all(|&(lo, hi)| lo < -0.9 && hi > 0.9));

            // Schedule metadata is deliberately varied; the retained map/alpha
            // and original graph/input remain the genuine completed objects.
            retained.alpha_config.deadline = Some(if expired {
                Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
            } else {
                authority + Duration::from_secs(60)
            });
            let verifier = BetaCrownVerifier::new(cfg);
            let evaluated = super::super::compute_root_objective_bounds(
                &verifier,
                &graph,
                &input,
                &objectives,
                &[0.0, 0.0],
                false,
                None,
                &retained,
                Some(authority),
                None,
                false,
                false,
            )
            .expect("fresh objective pass uses live original authority");
            assert_eq!(
                counter(),
                1,
                "no alpha/reference recollection after bootstrap"
            );
            assert_eq!(
                bits(&evaluated.initial_output),
                expected_output,
                "owned original output reused"
            );
            for &(lo, hi) in &evaluated.initial_obj_bounds {
                assert!(lo <= 0.0 && hi >= 0.0);
                assert!(
                    lo > -1.0e-4 && hi < 1.0e-4,
                    "fresh C fold must recover identical-output correlation: [{lo},{hi}]"
                );
            }
            assert_eq!(
                bits(&retained.initial_node_bounds["output"]),
                expected_output
            );
        }
    }

    #[ntest::timeout(30000)]
    #[test]
    fn retained_projection_preserves_order_and_interval_direction() {
        let (graph, input) = fixture();
        let authority = Instant::now() + Duration::from_secs(60);
        let cfg = config(authority);
        let retained = bootstrap(&graph, &input, &cfg);
        let objectives = vec![vec![1.0_f32, 0.0], vec![-2.0, 0.0], vec![1.0, -1.0]];
        let (output, projected) = project(&graph, &retained, &objectives, Some(authority)).unwrap();
        assert_eq!(bits(&output), bits(&retained.initial_node_bounds["output"]));
        assert_eq!(projected.len(), 3);
        assert!(
            projected[0].0 <= 0.0
                && projected[0].1 >= 1.0
                && projected[0].0 > -1.0e-4
                && projected[0].1 < 1.0001
        );
        assert!(
            projected[1].0 <= -2.0
                && projected[1].1 >= 0.0
                && projected[1].0 > -2.0001
                && projected[1].1 < 1.0e-4
        );
        assert!(
            projected[2].0 <= -1.0
                && projected[2].1 >= 1.0
                && projected[2].0 > -1.0001
                && projected[2].1 < 1.0001
        );
    }

    #[ntest::timeout(30000)]
    #[test]
    fn bounded_reference_tightening_expiry_keeps_one_map_and_fresh_objective_authority() {
        let (graph, input) = fixture();
        let authority = Instant::now() + Duration::from_secs(60);
        let cfg = config(authority);
        let counter = GraphNetwork::root_fixed_reference_collection_counter_for_test();
        let retained = compute_graph_bab_bootstrap_with_bounded_fixed_reference(
            &graph,
            &input,
            &cfg,
            None,
            Instant::now() + Duration::from_secs(30),
            authority,
            Duration::from_nanos(1),
        )
        .expect("exhausted optional tightening must retain foundational bounds");
        assert!(retained.retained_fixed_reference);
        assert_eq!(counter(), 1);
        let verifier = BetaCrownVerifier::new(cfg);
        let evaluated = super::super::compute_root_objective_bounds(
            &verifier,
            &graph,
            &input,
            &[vec![1.0_f32, -1.0]],
            &[0.0],
            false,
            None,
            &retained,
            Some(authority),
            None,
            false,
            false,
        )
        .expect("fresh correlated objective under live outer authority");
        assert_eq!(counter(), 1, "no second reference or alpha collection");
        let (lo, hi) = evaluated.initial_obj_bounds[0];
        assert!(lo <= 0.0 && hi >= 0.0 && lo > -1.0e-4 && hi < 1.0e-4);
    }

    #[ntest::timeout(30000)]
    #[test]
    fn reference_only_bootstrap_without_alpha_nodes_reaches_fresh_objectives() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "output",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32], [1.0]]), None).unwrap()),
        ));
        graph.set_output("output");
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        let authority = Instant::now() + Duration::from_secs(60);
        let cfg = config(authority);
        let counter = GraphNetwork::root_fixed_reference_collection_counter_for_test();
        let mut retained = bootstrap(&graph, &input, &cfg);
        assert!(retained.retained_fixed_reference);
        assert!(
            retained.phase_cap_optimizer_updates.is_none(),
            "reference-only is not a checkpoint"
        );
        assert!(retained
            .root_alpha_state
            .as_ref()
            .unwrap()
            .alphas
            .is_empty());
        assert_eq!(counter(), 1);
        retained.alpha_config.deadline =
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
        let verifier = BetaCrownVerifier::new(cfg);
        let evaluated = super::super::compute_root_objective_bounds(
            &verifier,
            &graph,
            &input,
            &[vec![1.0_f32, -1.0], vec![-1.0, 1.0]],
            &[0.0, 0.0],
            false,
            None,
            &retained,
            Some(authority),
            None,
            false,
            false,
        )
        .expect("easy affine graph needs no alpha optimizer or recollection");
        assert_eq!(counter(), 1);
        for &(lo, hi) in &evaluated.initial_obj_bounds {
            assert!(lo <= 0.0 && hi >= 0.0 && lo > -1.0e-4 && hi < 1.0e-4);
        }
    }

    #[ntest::timeout(30000)]
    #[test]
    fn retained_projection_rejects_missing_mismatched_or_expired_authority() {
        let (graph, input) = fixture();
        let authority = Instant::now() + Duration::from_secs(60);
        let cfg = config(authority);
        let mut retained = bootstrap(&graph, &input, &cfg);
        let objectives = vec![vec![1.0_f32, -1.0]];
        assert!(matches!(
            project(&graph, &retained, &objectives, None),
            Err(NyError::InvalidSpec(_))
        ));
        assert!(matches!(
            project(&graph, &retained, &[vec![1.0_f32]], Some(authority)),
            Err(NyError::ShapeMismatch { .. })
        ));
        retained.initial_node_bounds.remove("output");
        assert!(matches!(
            project(&graph, &retained, &objectives, Some(authority)),
            Err(NyError::InvalidSpec(_))
        ));
        // Deadline precedence proves no output lookup/projection is attempted
        // when the original authority has expired, even with a missing map.
        assert!(matches!(
            project(
                &graph,
                &retained,
                &objectives,
                Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
            ),
            Err(NyError::DeadlineExceeded(_))
        ));
    }

    #[ntest::timeout(30000)]
    #[test]
    fn retained_objective_wrong_width_propagates_source_error() {
        let (graph, input) = fixture();
        let authority = Instant::now() + Duration::from_secs(60);
        let cfg = config(authority);
        let retained = bootstrap(&graph, &input, &cfg);
        let verifier = BetaCrownVerifier::new(cfg);
        let result = super::super::compute_root_objective_bounds(
            &verifier,
            &graph,
            &input,
            &[vec![1.0_f32, 0.0, 0.0]],
            &[0.0],
            false,
            None,
            &retained,
            Some(authority),
            None,
            false,
            false,
        );
        assert!(
            matches!(
                result,
                Err(NyError::ShapeMismatch { .. }) | Err(NyError::InvalidSpec(_))
            ),
            "malformed C must not be projected/retried as a valid objective"
        );
    }

    #[ntest::timeout(30000)]
    #[test]
    fn retained_objective_expired_authority_precedes_spec_build_and_output_lookup() {
        let (graph, input) = fixture();
        let cfg = config(Instant::now() + Duration::from_secs(60));
        let mut retained = bootstrap(&graph, &input, &cfg);
        retained.initial_node_bounds.clear();
        let verifier = BetaCrownVerifier::new(cfg);
        let prior = super::super::root_objective_spec_build_count_for_test();
        let result = super::super::compute_root_objective_bounds(
            &verifier,
            &graph,
            &input,
            &[vec![1.0_f32, -1.0]],
            &[0.0],
            false,
            None,
            &retained,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
            None,
            false,
            false,
        );
        assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
        assert_eq!(
            super::super::root_objective_spec_build_count_for_test(),
            prior
        );
    }

    #[test]
    fn retained_fallback_error_classes_remain_narrow() {
        for error in [
            NyError::DeadlineExceeded("local only".into()),
            NyError::UnsupportedConfiguration("Constrained CROWN: cooperative finite test".into()),
            NyError::UnsupportedConfiguration("finite constrained IBP fallback test".into()),
        ] {
            assert!(may_reuse_after(&error));
        }
        for error in [
            NyError::NumericalInstability("NaN".into()),
            NyError::SoundnessRefusal("proof policy".into()),
            NyError::ShapeMismatch {
                expected: vec![2],
                got: vec![3],
            },
            NyError::InvalidSpec("source".into()),
            NyError::InvalidConfig("config".into()),
            NyError::InternalError("invariant".into()),
            NyError::UnsupportedOp("deadline word is not authority".into()),
            NyError::UnsupportedLayer("missing".into()),
            NyError::UnsupportedConfiguration("unknown mode".into()),
            NyError::InfeasibleDomain("not a numerical fallback".into()),
            NyError::CpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
                site: "unselected resource class",
            },
            NyError::GpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
            },
            NyError::GpuBatchCapacityExceeded {
                requested: 2,
                capacity: 1,
                unit: "rows",
                site: "unselected resource class",
            },
            NyError::LayerError {
                layer_index: 0,
                layer_type: "wrapped".into(),
                source: Box::new(NyError::DeadlineExceeded(
                    "not broad recursive matching".into(),
                )),
            },
        ] {
            assert!(!may_reuse_after(&error), "must propagate {error}");
        }
    }
}
