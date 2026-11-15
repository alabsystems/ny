// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::margin_row::bab::build_pack;
use crate::margin_row::head_group_source::HeadGroupSourceGuard;
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::RootGates;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};
use std::sync::atomic::{AtomicUsize, Ordering};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

fn six_spec(bias: f64) -> TwinSpec {
    let mut identity = vec![0.0; 36];
    for i in 0..6 {
        identity[i * 6 + i] = 1.0;
    }
    let mut head = vec![0.0; 42];
    for i in 0..6 {
        head[i * 6 + i] = 1.0;
        head[36 + i] = 1.0;
    }
    let mut final_weights = vec![0.0; 14];
    final_weights[..6].fill(1.0);
    final_weights[6] = -3.0 / 8.0;
    TwinSpec {
        n_in: 6,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: identity,
                bias: vec![2.0; 6],
                bias_err: vec![0.0; 6],
                weight_rel_err: 0.0,
                kernel: (6, 6, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (6, 1, 1),
                oshape: (6, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: head,
                bias: vec![-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -5.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (7, 6),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: final_weights,
                bias: vec![bias, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 7),
            },
        ],
    }
}

struct AllSlopes {
    negate: bool,
    calls: AtomicUsize,
}
impl HeadPhaseAdvice for AllSlopes {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        true
    }
    fn propose_functional(
        &self,
        r: &HeadFunctionalLpRequest,
        d: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if !r.validate() || !live(d) {
            return None;
        }
        assert_eq!(r.feature_count, 6);
        let sign = if self.negate { -1.0 } else { 1.0 };
        let mut row_duals = vec![0.0; r.rows.len()];
        for &row in r.slope_rows.iter().flatten() {
            let epigraph = r.rows[row].coefficients.last().unwrap().0;
            row_duals[row] = sign * r.objective[epigraph] * 3.0 / 8.0;
        }
        for &row in &r.phase_rows {
            let active = r.rows[row]
                .coefficients
                .first()
                .is_some_and(|(_, a)| *a > 0.0);
            row_duals[row] = sign * if active { 5.0 / 8.0 } else { 3.0 / 8.0 };
        }
        Some(HeadFunctionalLpHint { row_duals })
    }
}

#[test]
fn all_head_slopes_close_the_actual_six_dimensional_fixed_remainder_gap() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(43.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    // Independent source theorem: rho(x_i)>=3*x_i/8. The stable head is
    // 7+sum x_i, so the exact minimum is 1/16, attained at all x_i=0.
    assert_eq!(43.0 / 16.0 - 3.0 * 7.0 / 8.0, 1.0 / 16.0);
    let (actual_slopes, _) = native::source_remainder(&source, &[], deadline()).unwrap();
    assert_eq!(
        actual_slopes,
        baseline_slopes(&source, &[], deadline()).unwrap()
    );
    let old = native::replay_functionals(
        &eng,
        &pack,
        &source,
        &[],
        &[FunctionalProposal {
            mask: 0,
            slopes: actual_slopes.clone(),
            phase: vec![],
        }],
        deadline(),
    )
    .unwrap();
    assert!(
        old.bounds[0].1 <= 0.0,
        "real producer slopes {actual_slopes:?} already close the fixture"
    );
    let groups = source_groups(&source, 2, deadline()).unwrap();
    assert!(groups.iter().any(|g| g.len() == 4));
    for group in groups {
        let mut cover =
            propose_and_check(&source, &group, deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
        native::refine_complete_cover(&eng, &pack, &source, &group, &mut cover, deadline())
            .unwrap();
        assert!(
            cover.lower <= 0.0,
            "actual old/native group {group:?} unexpectedly closes"
        );
    }
    for negate in [false, true] {
        let provider = Arc::new(AllSlopes {
            negate,
            calls: AtomicUsize::new(0),
        });
        let backend: Arc<dyn HeadPhaseAdvice> = provider.clone();
        let output = refine_unconditional_with_provider(
            &eng,
            &pack,
            &source,
            &mut AdviceBudget::new(),
            deadline(),
            || Some(Arc::clone(&backend)),
        )
        .unwrap();
        let lower = output.lower.unwrap();
        assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
        assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
        assert_eq!(output.report.calls, 1);
        assert!(output.report.native_passes > 0);
    }
}

#[test]
fn all_head_replay_keeps_negative_source_and_full_unselected_q_intercepts() {
    let _capture = HeadGroupSourceGuard::enter(true);
    for bias in [41.0 / 16.0, 43.0 / 16.0] {
        let net = TwinNet::compile(&six_spec(bias)).unwrap();
        let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let pack = build_pack(&eng, None, &[]).unwrap();
        let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
        bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
        let mut slopes = vec![3.0 / 8.0; 7];
        slopes[6] = -3.0 / 8.0;
        let p = FunctionalProposal {
            mask: 0,
            slopes,
            phase: vec![],
        };
        let out = native::replay_functionals(
            &eng,
            &pack,
            &source,
            &[],
            std::slice::from_ref(&p),
            deadline(),
        )
        .unwrap();
        assert!(out.bounds[0].1 <= bias - 21.0 / 8.0);
        if bias < 21.0 / 8.0 {
            assert!(out.bounds[0].1 < 0.0);
        }
        for q in &mut source.q[..6] {
            q.lo = 1.0 / 8.0;
            q.hi = 1.0;
        }
        let out = native::replay_functionals(&eng, &pack, &source, &[], &[p], deadline()).unwrap();
        assert!(
            out.bounds[0].1 < 0.0,
            "full Q/Y intercepts must defeat nominal all-head cancellation"
        );
    }
}

#[test]
fn selected_functional_replay_retains_full_q_phase_residual_and_complete_masks() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(43.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let group = [0, 1];
    let mut cover = propose_and_check(&source, &group, deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    let before = cover.phase_lower.clone();
    let backend: Arc<dyn HeadPhaseAdvice> = Arc::new(AllSlopes {
        negate: false,
        calls: AtomicUsize::new(0),
    });
    let report = refine_cover_with_provider(
        &eng,
        &pack,
        &source,
        &group,
        &mut cover,
        &mut AdviceBudget::new(),
        deadline(),
        || Some(Arc::clone(&backend)),
    )
    .unwrap();
    assert_eq!(cover.phases, 4);
    assert_eq!(cover.phase_lower.len(), 4);
    assert!(cover.phase_lower.iter().zip(&before).all(|(a, b)| a >= b));
    assert!(report.improved_phases > 0);
    assert!(cover.lower > 1.0 / 32.0 && cover.lower <= 1.0 / 16.0);
    drop(cover);
    source.q[0] = I {
        lo: 1.0 / 8.0,
        hi: 1.0,
    };
    let mut slopes = vec![3.0 / 8.0; 7];
    slopes[6] = -3.0 / 8.0;
    let p = FunctionalProposal {
        mask: 3,
        slopes,
        phase: vec![3.0 / 16.0, 5.0 / 8.0],
    };
    // For selected head0 qmid=9/16, p=3/16 stores mu=3/8. Its
    // nonzero Delta remains priced on the complete original selected Y.
    let out = native::replay_functionals(&eng, &pack, &source, &group, &[p], deadline()).unwrap();
    assert!(out.bounds[0].1 < 0.0);
}

#[derive(Clone, Copy)]
enum Fault {
    Length,
    Nan,
    Excessive,
    Late,
    Panic,
    CapabilityPanic,
    Decline,
}
struct Faulty {
    fault: Fault,
    v1_calls: AtomicUsize,
    v2_calls: AtomicUsize,
}
impl HeadPhaseAdvice for Faulty {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        self.v1_calls.fetch_add(1, Ordering::Relaxed);
        None
    }
    fn supports_functional_advice(&self) -> bool {
        if matches!(self.fault, Fault::CapabilityPanic) {
            panic!("injected capability fault");
        }
        true
    }
    fn propose_functional(
        &self,
        r: &HeadFunctionalLpRequest,
        d: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        self.v2_calls.fetch_add(1, Ordering::Relaxed);
        let mut row_duals = vec![0.0; r.rows.len()];
        match self.fault {
            Fault::Length => {
                row_duals.pop();
            }
            Fault::Nan => row_duals[0] = f64::NAN,
            Fault::Excessive => row_duals[0] = 2e100,
            Fault::Late => {
                while live(d) {
                    std::thread::yield_now();
                }
            }
            Fault::Panic => panic!("injected functional fault"),
            Fault::CapabilityPanic | Fault::Decline => return None,
        }
        Some(HeadFunctionalLpHint { row_duals })
    }
}

#[test]
fn faulty_global_advice_preserves_old_masks_and_shares_the_v1_call_allowance() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(43.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    for fault in [
        Fault::Length,
        Fault::Nan,
        Fault::Excessive,
        Fault::Late,
        Fault::Panic,
        Fault::CapabilityPanic,
    ] {
        let backend: Arc<dyn HeadPhaseAdvice> = Arc::new(Faulty {
            fault,
            v1_calls: AtomicUsize::new(0),
            v2_calls: AtomicUsize::new(0),
        });
        let output = refine_unconditional_with_provider(
            &eng,
            &pack,
            &source,
            &mut AdviceBudget::new(),
            deadline(),
            || Some(Arc::clone(&backend)),
        )
        .unwrap();
        assert!(output.lower.is_none());
        let mut cover =
            propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
        let before = cover.phase_lower.clone();
        refine_cover_with_provider(
            &eng,
            &pack,
            &source,
            &[0, 1],
            &mut cover,
            &mut AdviceBudget::new(),
            deadline(),
            || Some(Arc::clone(&backend)),
        )
        .unwrap();
        assert_eq!(cover.phase_lower, before);
    }
    let provider = Arc::new(Faulty {
        fault: Fault::Decline,
        v1_calls: AtomicUsize::new(0),
        v2_calls: AtomicUsize::new(0),
    });
    let backend: Arc<dyn HeadPhaseAdvice> = provider.clone();
    let mut failed_export_budget = AdviceBudget::new();
    let failed_export = ask_with_elapsed(
        &eng,
        &pack,
        &source,
        &[0, 1],
        4,
        &mut failed_export_budget,
        deadline(),
        Instant::now(),
        || Some(Arc::clone(&backend)),
        || Duration::from_millis(29),
    )
    .unwrap();
    assert!(failed_export.hint.is_none());
    assert_eq!(failed_export.report.calls, 0);
    assert_eq!(failed_export_budget.charged, Duration::from_millis(29));
    assert_eq!(provider.v2_calls.load(Ordering::Relaxed), 0);
    let mut budget = AdviceBudget::new();
    refine_unconditional_with_provider(&eng, &pack, &source, &mut budget, deadline(), || {
        Some(Arc::clone(&backend))
    })
    .unwrap();
    let mut cover = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    advice::refine_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut budget,
        deadline(),
        || Some(Arc::clone(&backend)),
    )
    .unwrap();
    refine_cover_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut budget,
        deadline(),
        || Some(Arc::clone(&backend)),
    )
    .unwrap();
    assert_eq!(budget.calls, 2);
    assert_eq!(provider.v2_calls.load(Ordering::Relaxed), 1);
    assert_eq!(provider.v1_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn legacy_provider_expiry_source_mutation_and_incomplete_cover_refuse_global_work() {
    struct Legacy;
    impl HeadPhaseAdvice for Legacy {
        fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
            None
        }
    }
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(43.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let mut pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let mut budget = AdviceBudget::new();
    let backend: Arc<dyn HeadPhaseAdvice> = Arc::new(Legacy);
    let output =
        refine_unconditional_with_provider(&eng, &pack, &source, &mut budget, deadline(), || {
            Some(Arc::clone(&backend))
        })
        .unwrap();
    assert!(output.lower.is_none());
    assert_eq!(budget.calls, 0);
    assert!(budget.charged > Duration::ZERO);
    let out = refine_unconditional_with_provider(
        &eng,
        &pack,
        &source,
        &mut budget,
        Instant::now(),
        || panic!("expired source cannot construct provider"),
    )
    .unwrap();
    assert!(out.lower.is_none());
    let mut cover = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    cover.phase_lower.pop();
    assert!(refine_cover_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut budget,
        deadline(),
        || panic!("incomplete cover cannot construct provider")
    )
    .is_none());
    pack.al.b[0] += 0.25;
    assert!(refine_unconditional_with_provider(
        &eng,
        &pack,
        &source,
        &mut budget,
        deadline(),
        || panic!("mutated source cannot construct provider")
    )
    .is_none());
}

#[test]
fn global_setup_faults_charge_measured_time_without_spending_legacy_call_slots() {
    struct Legacy;
    impl HeadPhaseAdvice for Legacy {
        fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
            None
        }
    }
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(43.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    for kind in 0..4 {
        let backend: Arc<dyn HeadPhaseAdvice> = if kind == 2 {
            Arc::new(Faulty {
                fault: Fault::CapabilityPanic,
                v1_calls: AtomicUsize::new(0),
                v2_calls: AtomicUsize::new(0),
            })
        } else {
            Arc::new(Legacy)
        };
        let mut budget = AdviceBudget::new();
        let begin = Instant::now();
        let result = ask_with_elapsed(
            &eng,
            &pack,
            &source,
            &[],
            0,
            &mut budget,
            deadline(),
            begin,
            || match kind {
                0 => None,
                1 => panic!("injected factory fault"),
                _ => Some(Arc::clone(&backend)),
            },
            || Duration::from_millis(17),
        )
        .unwrap();
        assert!(!result.supported && result.hint.is_none());
        assert!(result.baseline.is_some());
        assert_eq!(budget.calls, 0);
        assert_eq!(budget.charged, Duration::from_millis(17));
        // Both V1 slots remain available, but setup has reduced the shared
        // time allowance. This is checked with a synthetic clock, no sleep.
        budget.charge_elapsed(Duration::from_millis(190));
        assert_eq!(
            budget.call_deadline(begin, begin + Duration::from_secs(2)),
            Some(begin + Duration::from_millis(43))
        );
        budget.calls = 1;
        assert!(budget
            .call_deadline(begin, begin + Duration::from_secs(2))
            .is_some());
        budget.calls = 2;
        assert!(budget
            .call_deadline(begin, begin + Duration::from_secs(2))
            .is_none());
    }
    let mut invalid = source.clone();
    invalid.q[0].lo = f64::NAN;
    for (problem, expired, charged) in [(&invalid, false, 13), (&source, true, 5)] {
        let mut budget = AdviceBudget::new();
        let begin = Instant::now();
        let result = ask_with_elapsed(
            &eng,
            &pack,
            problem,
            &[],
            0,
            &mut budget,
            if expired { begin } else { deadline() },
            begin,
            || panic!("failed preparation cannot construct provider"),
            || Duration::from_millis(charged),
        )
        .unwrap();
        assert!(result.baseline.is_none() && result.hint.is_none());
        assert_eq!(budget.calls, 0);
        assert_eq!(budget.charged, Duration::from_millis(charged));
    }
    let provider = Arc::new(Faulty {
        fault: Fault::Decline,
        v1_calls: AtomicUsize::new(0),
        v2_calls: AtomicUsize::new(0),
    });
    let backend: Arc<dyn HeadPhaseAdvice> = provider.clone();
    let mut budget = AdviceBudget::new();
    budget.charge_elapsed(Duration::from_millis(17));
    let mut cover = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    for _ in 0..2 {
        advice::refine_with_provider(
            &eng,
            &pack,
            &source,
            &[0, 1],
            &mut cover,
            &mut budget,
            deadline(),
            || Some(Arc::clone(&backend)),
        )
        .unwrap();
    }
    assert_eq!(provider.v1_calls.load(Ordering::Relaxed), 2);
    assert_eq!(budget.calls, 2);
}

#[test]
fn projected_export_retains_explicit_sparse_roles_and_full_head_order() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(43.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let r = export_functional_lp(&eng, &source, &[], 0, deadline()).unwrap();
    assert!(r.validate());
    assert_eq!(r.feature_count, 6);
    assert_eq!(r.bounds.len(), 12);
    assert_eq!(
        r.slope_rows,
        vec![Some(1), Some(3), Some(5), Some(7), Some(9), Some(11), None]
    );
    assert_eq!((r.phase_rows).len(), 0);
    for &row in r.slope_rows.iter().flatten() {
        assert!((r.rows[row].lower + 2.0).abs() < 1e-10);
    }
    let p = export_functional_lp(&eng, &source, &[0, 1], 2, deadline()).unwrap();
    assert_eq!(p.phase_rows.len(), 2);
    assert_eq!(p.slope_rows[..2], [None, None]);
    assert_eq!(p.rows[p.phase_rows[0]].coefficients, vec![(0, -1.0)]);
    assert_eq!(p.rows[p.phase_rows[1]].coefficients, vec![(1, 1.0)]);
}

fn multi_spec(negative_third: bool) -> TwinSpec {
    let mut spec = six_spec(43.0 / 16.0);
    let target = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -3.0 / 8.0];
    let bias = 43.0 / 16.0;
    if let TwinOpSpec::Gemm {
        weight,
        bias: bs,
        shape,
        ..
    } = &mut spec.ops[5]
    {
        weight.clear();
        weight.extend_from_slice(&target);
        weight.extend_from_slice(&[0.0; 7]);
        weight.extend(target.iter().map(|v| -v));
        weight.extend(target.iter().map(|v| 0.5 * v));
        if negative_third {
            weight[21] = 7.0 / 8.0;
        }
        *bs = vec![bias, 0.0, -bias, 0.5 * bias];
        *shape = (4, 7);
    }
    spec
}

#[test]
fn two_global_hints_reach_three_margins_in_six_columns_before_group_search() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&multi_spec(false)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    let provider = Arc::new(AllSlopes {
        negate: false,
        calls: AtomicUsize::new(0),
    });
    let backend: Arc<dyn HeadPhaseAdvice> = provider.clone();
    let mut budget = AdviceBudget::new();
    let result = refine_portfolio_with_provider(
        &eng,
        &pack,
        &mut source,
        0,
        &[3, 1, 2],
        &[-0.5, -1.0, -2.0],
        &mut budget,
        deadline(),
        || Some(Arc::clone(&backend)),
    )
    .unwrap();
    for (lower, scale) in result.lower.iter().zip([0.5, 1.0, 2.0]) {
        let lower = lower.unwrap();
        assert!(lower > scale / 32.0 && lower <= scale / 16.0);
    }
    assert_eq!(provider.calls.load(Ordering::Relaxed), 2);
    assert_eq!(budget.calls, 2);
    assert_eq!(result.report.native_columns, 6);
    assert_eq!(result.report.native_passes, 2);
}

#[test]
fn warm_slopes_use_recipient_q_and_cannot_close_a_negative_recipient_source() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&multi_spec(true)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    let backend: Arc<dyn HeadPhaseAdvice> = Arc::new(AllSlopes {
        negate: true,
        calls: AtomicUsize::new(0),
    });
    let result = refine_portfolio_with_provider(
        &eng,
        &pack,
        &mut source,
        0,
        &[1, 2, 3],
        &[-1.0, -2.0, -0.5],
        &mut AdviceBudget::new(),
        deadline(),
        || Some(Arc::clone(&backend)),
    )
    .unwrap();
    assert!(result.lower[0].unwrap() > 0.0 && result.lower[1].unwrap() > 0.0);
    // Authored margin3 at x=(1,0,0,0,0,0) is -1/32.
    assert_eq!(43.0 / 32.0 + 1.0 / 8.0 - 3.0 * 8.0 / 16.0, -1.0 / 32.0);
    assert!(result.lower[2].unwrap() <= -1.0 / 32.0);
}

#[test]
fn prepared_global_columns_refuse_foreign_source_generation_and_expired_replay() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(43.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let mut slopes = vec![3.0 / 8.0; 7];
    slopes[6] = -3.0 / 8.0;
    let prepared = native::prepare_functionals(
        &eng,
        &pack,
        &source,
        &[],
        &[FunctionalProposal {
            mask: 0,
            slopes,
            phase: vec![],
        }],
        deadline(),
    )
    .unwrap();
    assert!(
        native::replay_prepared_functionals(&eng, &pack, &source, &prepared, Instant::now())
            .is_none()
    );
    // The same net object and declared box still produce a fresh private
    // build generation. Identical digests cannot rebase prepared columns.
    let next_root =
        RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let next_eng = BackwardEngine::new(&net, &next_root);
    let next_pack = build_pack(&next_eng, None, &[]).unwrap();
    let next_source = from_owned_pack(&next_eng, &next_pack, deadline()).unwrap();
    let snapshot = pack.head_source_snapshot(&eng).unwrap();
    let next_snapshot = next_pack.head_source_snapshot(&next_eng).unwrap();
    assert!(!snapshot.same_snapshot(&next_snapshot));
    assert!(native::replay_prepared_functionals(
        &next_eng,
        &next_pack,
        &next_source,
        &prepared,
        deadline()
    )
    .is_none());
    let other_net = TwinNet::compile(&six_spec(41.0 / 16.0)).unwrap();
    let other_root =
        RootGates::build(&other_net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let other_eng = BackwardEngine::new(&other_net, &other_root);
    let other_pack = build_pack(&other_eng, None, &[]).unwrap();
    let other = from_owned_pack(&other_eng, &other_pack, deadline()).unwrap();
    assert!(native::replay_prepared_functionals(
        &other_eng,
        &other_pack,
        &other,
        &prepared,
        deadline()
    )
    .is_none());
}

#[test]
fn wide_projected_geometry_reads_only_sampled_features_and_expands_support() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let width = 256;
    let spec = TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0; width],
                bias: vec![2.0; width],
                bias_err: vec![0.0; width],
                weight_rel_err: 0.0,
                kernel: (width, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (width, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0 / width as f64; 2 * width],
                bias: vec![-2.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, width),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, -0.5, 0.0, 0.0],
                bias: vec![0.25, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    };
    let mut net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let original = export_functional_lp(&eng, &source, &[], 0, deadline()).unwrap();
    assert_eq!(original.feature_count, 128);
    assert_eq!(original.bounds.len(), 129);
    assert!(original.bounds[..128]
        .iter()
        .all(|b| b[0] < 1.0 && b[1] > 3.0));
    let row = original.slope_rows[0].unwrap();
    assert_eq!(original.rows[row].coefficients.len(), 129);
    assert!((original.rows[row].lower + 1.0).abs() < 1e-10);
    assert!(original.bounds[128][1] >= source.head[0].y.hi);
    drop(eng);
    // Feature1 is omitted by the fixed uniform128-of256 sample. Corrupting
    // only that search matrix entry leaves the advice snapshot identical,
    // demonstrating the exporter does not traverse omitted coefficients.
    if let TwinOp::Gemm { weight, .. } = &mut net.ops[3] {
        weight[1] = f64::NAN;
    } else {
        panic!("fixture head must be Gemm");
    }
    let changed_eng = BackwardEngine::new(&net, &root);
    let changed = export_functional_lp(&changed_eng, &source, &[], 0, deadline()).unwrap();
    assert_eq!(
        serde_json::to_vec(&original).unwrap(),
        serde_json::to_vec(&changed).unwrap()
    );
    // These deliberately stale rows are NEVER accepted as authority: even
    // though the rank-only exporter ignores that omitted NaN, replay refuses.
    let slopes = baseline_slopes(&source, &[], deadline()).unwrap();
    assert!(native::replay_functionals(
        &changed_eng,
        &pack,
        &source,
        &[],
        &[FunctionalProposal {
            mask: 0,
            slopes,
            phase: vec![],
        }],
        deadline()
    )
    .is_none());
}
