// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dual-source adapter to the existing concurrent attack lifecycle.
//! A proxy reports only whether the unchanged proof decided. Its real result
//! and authorization stay in a separate slot and are the only proof output.
//! Candidate publication uses the existing original-model dual replay, the
//! shared attack door, confirmation latch, and terminal-publication claim.
use super::*;
use crate::commands::beta_crown::attack_concurrent as race;
use ny_propagate::{BabVerificationStatus, BetaCrownResult};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub(super) struct PublishTarget {
    pub(super) results_file: PathBuf,
    pub(super) deadline: Instant,
}

pub(super) fn admit(
    target: Option<&PublishTarget>,
    diff: &GraphNetwork,
    start: Instant,
    slice_deadline: Instant,
    overall_deadline: Instant,
) -> Option<race::AdmittedRace> {
    let site = race::RaceSite::RelationalShared;
    let planned = race::plan_window(
        start,
        Some(slice_deadline),
        Some(overall_deadline),
        race::proof_pool_width(),
        race::static_width(),
    );
    let facts = race::AdmissionFacts {
        publisher_armed: target.is_some(),
        // The multi-network early route precedes the ordinary margin worker.
        margin_row_lane_live: false,
        // forward_point_vec has no accelerator handle: this attack is CPU.
        shared_device: false,
        sign_net: race::graph_has_sign(diff),
        qdq_report_only: crate::commands::beta_crown::qdq_probe::qdq_relaxation_engaged(),
    };
    let window = match race::admit(&facts, planned) {
        Ok(window) => window,
        Err(route) => {
            race::print_admit_line(site, route, planned.as_ref().ok(), "na");
            return None;
        }
    };
    let (allowed, lease) = race::admit_memory();
    if !allowed {
        race::print_admit_line(site, race::AdmitRoute::LeaseDenied, Some(&window), lease);
        return None;
    }
    let Some(pool) = race::build_pool(window.width_pool) else {
        crate::router::lease::release(ny_core::router::lease::LeaseHolder::AttackConcurrent);
        race::print_admit_line(
            site,
            race::AdmitRoute::PoolBuildFailed,
            Some(&window),
            lease,
        );
        return None;
    };
    race::print_admit_line(site, race::AdmitRoute::Spawned, Some(&window), lease);
    eprintln!(
        "[relational-overlap] armed source=preset serial_slice_ms={} parallel_window_ms={} \
         host_threads={} attack_threads={} original_deadline=1",
        window.slice.as_millis(),
        window.window.as_millis(),
        window.host_threads,
        window.width_pool,
    );
    Some(race::AdmittedRace { window, pool })
}

struct SourcePair {
    f: GraphNetwork,
    g: GraphNetwork,
    onnx_f: PathBuf,
    onnx_g: PathBuf,
    dual: DualNetworkSpec,
    epsilon: f64,
}

impl SourcePair {
    fn revalidate(&self, point: &[f32]) -> Option<String> {
        revalidate_isomorphic_witness(
            &self.f,
            &self.g,
            &self.onnx_f,
            &self.onnx_g,
            point,
            &self.dual,
            self.epsilon,
        )
    }
}

fn prepare(
    source: Arc<SourcePair>,
    target: PublishTarget,
    candidate: Vec<f32>,
    report: race::ConcurrentSatReport,
) -> race::PreparedSat {
    use race::ConcurrentSatPublication as Declined;
    // Registration precedes spawning the detached replay, as in the existing
    // single-model adapter. The main publication cannot miss a prepared job.
    let Some(entered) = AttackPublicationGuard::enter_in(&SAT_PRECEDENCE, &ATTACK_DOOR_CHANGED)
    else {
        return race::PreparedSat::Declined(Declined::DoorClosed);
    };
    race::PreparedSat::Adjudicate(Box::new(move || {
        let _entered = entered;
        if Instant::now() >= target.deadline {
            return Declined::SuppressedAtDeadline;
        }
        if crate::commands::beta_crown::qdq_probe::qdq_relaxation_engaged() {
            return Declined::Withheld;
        }
        let _adjudication = enter_sat_adjudication();
        let Some(witness) = source.revalidate(&candidate) else {
            return Declined::Downgraded;
        };
        // Only the unchanged original dual-ORT gate can reach this latch.
        note_confirmed_sat();
        if Instant::now() >= target.deadline {
            return Declined::SuppressedAtDeadline;
        }
        let result = VnncompResult::Sat {
            witness: Some(witness),
        };
        let outcome = publish_attack_sat_in(&SAT_PRECEDENCE, &TERMINAL_PUBLICATION_CLAIMED, || {
            if Instant::now() >= target.deadline {
                anyhow::bail!(
                    "relational candidate reached publication after its absolute deadline"
                );
            }
            write_results(&target.results_file, &result)
        });
        match outcome {
            AttackSatPublication::Published => {
                let line = report.published_line();
                eprintln!("{line}");
                println!("Result: {}", result.token());
                crate::flight::note(
                    "result_publish",
                    crate::flight::FlightStatus::Ran,
                    Some(result.token().into()),
                );
                crate::flight::finish(result.token());
                crate::flight::global().write_sidecar(&target.results_file);
                close_proof_bundle(&result);
                exit_scored_instance_without_teardown();
            }
            AttackSatPublication::RefusedUnconfirmed => Declined::Unconfirmed,
            AttackSatPublication::RefusedClaimTaken => Declined::ClaimTaken,
            AttackSatPublication::RefusedDoorClosed => Declined::DoorClosed,
            AttackSatPublication::WriteFailed(error) => Declined::WriteFailed(error),
        }
    }))
}

fn proxy(result: &Result<VnncompResult>) -> BetaCrownResult {
    let status = match result {
        Ok(VnncompResult::Unsat(_)) => BabVerificationStatus::Verified,
        Ok(VnncompResult::Sat { .. }) => BabVerificationStatus::Violated {
            counterexample: Vec::new(),
            output: Vec::new(),
        },
        _ => BabVerificationStatus::Timeout,
    };
    BetaCrownResult {
        result: status,
        domains_explored: 0,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: Duration::ZERO,
        output_bounds: None,
    }
}

/// Candidate-only worker shared by the actual scoped race and its controls.
/// Every returned point still passes the unchanged original dual-source gate
/// before publication. The prepared route has no proof authority.
#[allow(clippy::too_many_arguments)]
fn attack_candidate(
    diff: &GraphNetwork,
    graph_f: &GraphNetwork,
    graph_g: &GraphNetwork,
    search_bounds: &[Bound],
    epsilon: f64,
    deadline: Instant,
    token: &ny_propagate::PgdCancelToken,
    point_vjp: bool,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if token.is_cancelled() || Instant::now() >= deadline {
        return None;
    }
    let advice = if point_vjp {
        eprintln!("[relational-vjp] requested source=preset original_deadline=1");
        let advice = relational_vjp::DenseTwinAdvice::build_cancellable(
            graph_f,
            graph_g,
            search_bounds,
            deadline,
            Some(token),
        );
        if advice.is_some() {
            eprintln!("[relational-vjp] armed source=preset device=cpu grid_random_coordinate_pgd_caps_unchanged=1 analytic_template_gradient=1");
        } else {
            eprintln!(
                "[relational-vjp] declined reason=unsupported_or_stopped_original_tower_template"
            );
        }
        advice
    } else {
        None
    };
    let candidate = if point_vjp {
        search_isomorphic_deviation_with_advice_and_cancel(
            diff,
            search_bounds,
            epsilon,
            deadline,
            advice.as_ref(),
            Some(token),
        )
    } else {
        search_isomorphic_deviation_cancellable(diff, search_bounds, epsilon, deadline, Some(token))
    };
    if let Some(advice) = &advice {
        advice.receipt();
    }
    let candidate = candidate?;
    if token.is_cancelled() || Instant::now() >= deadline {
        return None;
    }
    let output = forward_point_vec(diff, &candidate).ok()?;
    Some((candidate, output))
}

/// Scheduling markers cannot authorize a result. Only the held original
/// result, a source-validated candidate, or the unconsumed proof may return.
fn arbitrate<V, P>(
    raced: Result<BetaCrownResult>,
    actual: Option<Result<VnncompResult>>,
    validate: V,
    fallback: P,
) -> (Result<VnncompResult>, bool)
where
    V: FnOnce(&[f32]) -> Option<String>,
    P: FnOnce() -> Result<VnncompResult>,
{
    if matches!(actual, Some(Ok(VnncompResult::Sat { .. }))) {
        return (actual.expect("matched actual result"), false);
    }
    if let Ok(BetaCrownResult {
        result: BabVerificationStatus::Violated { counterexample, .. },
        ..
    }) = raced
    {
        if let Some(witness) = validate(&counterexample) {
            return (
                Ok(VnncompResult::Sat {
                    witness: Some(witness),
                }),
                true,
            );
        }
    }
    (actual.unwrap_or_else(fallback), false)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<P>(
    admitted: race::AdmittedRace,
    target: PublishTarget,
    diff: &GraphNetwork,
    graph_f: &GraphNetwork,
    graph_g: &GraphNetwork,
    onnx_f: &Path,
    onnx_g: &Path,
    dual: &DualNetworkSpec,
    epsilon: f64,
    start: Instant,
    slice_deadline: Instant,
    overall_deadline: Instant,
    point_vjp: bool,
    proof: P,
) -> Result<VnncompResult>
where
    P: FnOnce() -> Result<VnncompResult>,
{
    let source = Arc::new(SourcePair {
        f: graph_f.clone(),
        g: graph_g.clone(),
        onnx_f: onnx_f.to_owned(),
        onnx_g: onnx_g.to_owned(),
        dual: dual.clone(),
        epsilon,
    });
    let search_bounds: Vec<Bound> = dual
        .f_input_bounds
        .iter()
        .map(|&b| inward_bound(b))
        .collect();
    let publish_source = Arc::clone(&source);
    let publish = move |candidate: &BetaCrownResult, report: &race::ConcurrentSatReport| {
        let BabVerificationStatus::Violated { counterexample, .. } = &candidate.result else {
            return race::PreparedSat::Declined(race::ConcurrentSatPublication::Downgraded);
        };
        prepare(
            Arc::clone(&publish_source),
            target.clone(),
            counterexample.clone(),
            *report,
        )
    };
    let proof_body = Mutex::new(Some(proof));
    let proof_result = Mutex::new(None);
    let run_proof = || {
        let body = proof_body
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .expect("proof runs once");
        let actual = body();
        let marker = proxy(&actual);
        *proof_result.lock().unwrap_or_else(|p| p.into_inner()) = Some(actual);
        Ok(marker)
    };
    let attack = |token: ny_propagate::PgdCancelToken, deadline: Instant| {
        let Some((candidate, output)) = attack_candidate(
            diff,
            &source.f,
            &source.g,
            &search_bounds,
            epsilon,
            deadline,
            &token,
            point_vjp,
        ) else {
            return Ok(None);
        };
        Ok(Some((
            ndarray::Array1::from(candidate).into_dyn(),
            ndarray::Array1::from(output).into_dyn(),
        )))
    };
    let (raced, _) = race::run_race_observed(
        race::RacePlan {
            site: race::RaceSite::RelationalShared,
            window: admitted.window,
            site_start: start,
            slice_deadline,
            overall_deadline: Some(overall_deadline),
            lane_start: start,
            publish: &publish,
        },
        admitted.pool,
        attack,
        run_proof,
    );
    let actual = proof_result.into_inner().unwrap_or_else(|p| p.into_inner());
    // Spawn failure can return an unadjudicated candidate without invoking
    // proof. On a failed dual replay, run the still-owned original closure,
    // with its original absolute deadline and remaining allowance.
    let (result, confirmed) = arbitrate(
        raced,
        actual,
        |point| {
            (Instant::now() < overall_deadline)
                .then(|| source.revalidate(point))
                .flatten()
        },
        || {
            proof_body
                .into_inner()
                .unwrap_or_else(|p| p.into_inner())
                .expect("proof remains owned")()
        },
    );
    if confirmed {
        note_confirmed_sat();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn marker(status: BabVerificationStatus) -> BetaCrownResult {
        BetaCrownResult {
            result: status,
            domains_explored: 0,
            domains_verified: 0,
            cuts_generated: 0,
            max_depth_reached: 0,
            time_elapsed: Duration::ZERO,
            output_bounds: None,
        }
    }
    fn candidate() -> Result<BetaCrownResult> {
        Ok(marker(BabVerificationStatus::Violated {
            counterexample: vec![0.25],
            output: vec![1.0],
        }))
    }

    #[test]
    fn verified_proxy_without_an_original_result_cannot_mint_unsat() {
        let calls = Cell::new(0);
        let (result, confirmed) = arbitrate(
            Ok(marker(BabVerificationStatus::Verified)),
            None,
            |_| panic!("verified marker has no candidate"),
            || {
                calls.set(calls.get() + 1);
                Ok(VnncompResult::Timeout)
            },
        );
        assert_eq!(result.unwrap(), VnncompResult::Timeout);
        assert!(!confirmed);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn rejected_candidate_retains_the_original_inconclusive_result() {
        let calls = Cell::new(0);
        let (result, confirmed) = arbitrate(
            candidate(),
            Some(Ok(VnncompResult::Timeout)),
            |point| {
                assert_eq!(point, &[0.25]);
                calls.set(calls.get() + 1);
                None
            },
            || panic!("original proof already ran"),
        );
        assert_eq!(result.unwrap(), VnncompResult::Timeout);
        assert!(!confirmed);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn only_a_validated_candidate_may_replace_an_inconclusive_proof() {
        let (result, confirmed) = arbitrate(
            candidate(),
            Some(Ok(VnncompResult::Unknown)),
            |point| {
                assert_eq!(point, &[0.25]);
                Some("validated original dual witness".into())
            },
            || panic!("original proof already ran"),
        );
        assert_eq!(
            result.unwrap(),
            VnncompResult::Sat {
                witness: Some("validated original dual witness".into())
            }
        );
        assert!(confirmed);
    }

    #[test]
    fn original_proof_sat_keeps_its_original_witness() {
        let (result, confirmed) = arbitrate(
            candidate(),
            Some(Ok(VnncompResult::Sat {
                witness: Some("proof witness".into()),
            })),
            |_| panic!("already source-validated proof witness wins"),
            || panic!("proof ran"),
        );
        assert_eq!(
            result.unwrap(),
            VnncompResult::Sat {
                witness: Some("proof witness".into())
            }
        );
        assert!(!confirmed);
    }

    #[test]
    fn original_errors_and_unconsumed_proof_fallback_are_preserved() {
        let (result, confirmed) = arbitrate(
            Ok(marker(BabVerificationStatus::Verified)),
            Some(Err(anyhow!("original proof error"))),
            |_| panic!("no candidate"),
            || panic!("proof already ran"),
        );
        assert_eq!(result.unwrap_err().to_string(), "original proof error");
        assert!(!confirmed);
        let called = Cell::new(0);
        let (result, confirmed) = arbitrate(
            candidate(),
            None,
            |_| None,
            || {
                called.set(called.get() + 1);
                Ok(VnncompResult::Unknown)
            },
        );
        assert_eq!(result.unwrap(), VnncompResult::Unknown);
        assert!(!confirmed);
        assert_eq!(called.get(), 1);
    }

    #[test]
    fn typed_relational_projection_is_opt_in_and_preserves_the_legacy_default() {
        let parse = |text| serde_yaml::from_str::<crate::preset::PresetConfig>(text).unwrap();
        assert!(!RelationalPresetKeys::default().attack_concurrent);
        assert!(!RelationalPresetKeys::from_preset(&parse("{}\n")).attack_concurrent);
        assert!(
            RelationalPresetKeys::from_preset(&parse("attack:\n  concurrent: true\n"))
                .attack_concurrent
        );
        assert!(
            !RelationalPresetKeys::from_preset(&parse("attack:\n  concurrent: false\n"))
                .attack_concurrent
        );
    }

    #[test]
    fn existing_window_arithmetic_keeps_the_original_overall_deadline() {
        let start = Instant::now();
        let slice = start + Duration::from_secs(30);
        let window = race::plan_window(
            start,
            Some(slice),
            Some(start + Duration::from_secs(95)),
            4,
            4,
        )
        .unwrap();
        assert_eq!(window.slice, Duration::from_secs(30));
        assert_eq!(window.window, Duration::from_secs(60));
        assert!(race::plan_window(
            start,
            Some(slice),
            Some(start + Duration::from_secs(64)),
            4,
            4
        )
        .is_err());
        assert!(admit(
            None,
            &GraphNetwork::new(),
            start,
            slice,
            start + Duration::from_secs(95)
        )
        .is_none());
    }

    #[test]
    fn cancelled_shared_search_preserves_a_planted_violation_control() {
        use ny_propagate::layers::LinearLayer;
        use ny_propagate::{GraphNode, Layer};
        let mut diff = GraphNetwork::new();
        diff.add_node(GraphNode::from_input(
            "constant",
            Layer::Linear(LinearLayer::new(ndarray::arr2(&[[0.0]]), Some(arr1(&[1.0]))).unwrap()),
        ));
        diff.set_output("constant");
        let bounds = [Bound::new(-1.0, 1.0)];
        let active = ny_propagate::PgdCancelToken::new();
        let point = search_isomorphic_deviation_cancellable(
            &diff,
            &bounds,
            0.5,
            Instant::now() + Duration::from_secs(10),
            Some(&active),
        )
        .expect("planted constant violation");
        assert!(forward_point_vec(&diff, &point).unwrap()[0] > 0.5);
        let cancelled = ny_propagate::PgdCancelToken::new();
        cancelled.cancel();
        assert!(search_isomorphic_deviation_cancellable(
            &diff,
            &bounds,
            0.5,
            Instant::now() + Duration::from_secs(10),
            Some(&cancelled)
        )
        .is_none());
    }

    fn constant_difference_pair() -> (GraphNetwork, GraphNetwork, GraphNetwork) {
        use ndarray::{arr1, arr2};
        use ny_propagate::layers::LinearLayer;
        let tower = |bias| {
            let mut graph = GraphNetwork::new();
            let layer = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[bias]))).unwrap();
            graph.add_node(GraphNode::from_input("output", Layer::Linear(layer)));
            graph.set_output("output");
            graph
        };
        let f = tower(0.25);
        let g = tower(0.0);
        let diff = build_difference_network(&f, &g).unwrap();
        (f, g, diff)
    }

    #[test]
    fn actual_scoped_worker_uses_prepared_advice_and_replays_original_graphs() {
        let (f, g, diff) = constant_difference_pair();
        let token = ny_propagate::PgdCancelToken::new();
        relational_vjp::test_reset_work();
        let (point, output) = attack_candidate(
            &diff,
            &f,
            &g,
            &[Bound::new(-1.0, 1.0)],
            0.05,
            Instant::now() + Duration::from_secs(10),
            &token,
            true,
        )
        .expect("planted violation from actual worker helper");
        let (probes, forwards) = relational_vjp::test_work();
        assert_eq!(probes, 2);
        assert!(forwards > 0, "actual prepared forwards must execute");
        let original_f = forward_point_vec(&f, &point).unwrap();
        let original_g = forward_point_vec(&g, &point).unwrap();
        assert!(original_f
            .iter()
            .zip(&original_g)
            .any(|(f, g)| (f - g).abs() > 0.05));
        assert_eq!(output, forward_point_vec(&diff, &point).unwrap());
    }

    #[test]
    fn actual_worker_keeps_legacy_and_pre_cancelled_paths_without_template_work() {
        let (f, g, diff) = constant_difference_pair();
        let token = ny_propagate::PgdCancelToken::new();
        relational_vjp::test_reset_work();
        assert!(attack_candidate(
            &diff,
            &f,
            &g,
            &[Bound::new(-1.0, 1.0)],
            0.05,
            Instant::now() + Duration::from_secs(10),
            &token,
            false,
        )
        .is_some());
        assert_eq!(relational_vjp::test_work(), (0, 0));
        token.cancel();
        for enabled in [false, true] {
            assert!(attack_candidate(
                &diff,
                &f,
                &g,
                &[Bound::new(-1.0, 1.0)],
                0.05,
                Instant::now() + Duration::from_secs(10),
                &token,
                enabled,
            )
            .is_none());
            assert_eq!(relational_vjp::test_work(), (0, 0));
        }
    }

    #[test]
    fn shipped_advice_policy_executes_worker_and_explicit_off_keeps_legacy_search() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
        let path = resolve_preset_path(&root, "isomorphic_acasxu_2026").unwrap();
        let snapshot = BetaCrownPresetSnapshot::load(&path);
        let shipped = snapshot.loaded().expect("valid shipped policy");
        let (f, g, diff) = constant_difference_pair();
        for enabled in [true, false] {
            let mut policy = (*shipped).clone();
            if !enabled {
                policy.attack.concurrent = Some(false);
                policy.attack.relational_point_vjp = Some(false);
            }
            let keys = RelationalPresetKeys::from_preset(&policy);
            assert_eq!(keys.attack_concurrent, enabled);
            assert_eq!(keys.relational_point_vjp, enabled);
            relational_vjp::test_reset_work();
            let token = ny_propagate::PgdCancelToken::new();
            let (point, output) = attack_candidate(
                &diff,
                &f,
                &g,
                &[Bound::new(-1.0, 1.0)],
                0.05,
                Instant::now() + Duration::from_secs(10),
                &token,
                keys.relational_point_vjp,
            )
            .expect("planted violation through the projected worker policy");
            let (probes, forwards) = relational_vjp::test_work();
            if enabled {
                assert_eq!(probes, 2);
                assert!(forwards > 0, "shipped policy must reach prepared CPU work");
            } else {
                assert_eq!(
                    (probes, forwards),
                    (0, 0),
                    "explicit off must retain the legacy worker"
                );
            }
            let original_f = forward_point_vec(&f, &point).unwrap();
            let original_g = forward_point_vec(&g, &point).unwrap();
            assert!(original_f
                .iter()
                .zip(original_g)
                .any(|(f, g)| (*f - g).abs() > 0.05));
            assert_eq!(output, forward_point_vec(&diff, &point).unwrap());
        }
    }
}
