use super::*;
use std::time::Duration;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}
fn row(a: f64, b: f64) -> Affine {
    Affine {
        a: vec![I::p(a)],
        b: I::p(b),
    }
}
fn env(a: f64, lb: f64, ub: f64) -> Envelope {
    Envelope {
        lower: row(a, lb),
        upper: row(a, ub),
        y: I {
            lo: lb - a.abs(),
            hi: ub + a.abs(),
        },
        alpha: 0.5,
    }
}
fn twins() -> SourceHeadProblem {
    SourceHeadProblem {
        root: vec![I { lo: -1.0, hi: 1.0 }],
        head: vec![env(1.0, 0.0, 0.0); 2],
        q: vec![I::p(1.0), I::p(-1.0)],
        bias: I::p(0.125),
    }
}
fn actual(problem: &SourceHeadProblem, values: &[f64]) -> f64 {
    problem.bias.lo
        + problem
            .q
            .iter()
            .zip(values)
            .map(|(q, y)| q.lo * y.max(0.0))
            .sum::<f64>()
}

#[test]
fn shared_source_group_closes_dyadic_cancellation_that_independent_triangles_lose() {
    let p = twins();
    let d = deadline();
    let bound = propose_and_check(&p, &[0, 1], d, MAX_PROPOSAL_ROUNDS).unwrap();
    assert_eq!(bound.phases, 4);
    assert!(bound.lower > 0.124999999);
    // Both independently selected alpha=.5 lines yield q0*rho(x) -
    // q1*upper_chord(x) >= -.5, before the exact +.125 margin bias.
    assert!(bound.lower > -0.375 + 0.49);
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        assert_eq!(actual(&p, &[x, x]), 0.125);
        assert!(bound.lower <= actual(&p, &[x, x]));
    }
}

#[test]
fn envelopes_are_relaxations_not_authored_affine_equalities() {
    let mut p = twins();
    p.head = vec![env(1.0, 0.0, 0.0625); 2];
    let bound = propose_and_check(&p, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    assert!(bound.lower > 0.06249999 && bound.lower <= 0.0625);
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        let nonlinear = x + x * x / 16.0;
        assert!(bound.lower <= actual(&p, &[nonlinear, nonlinear]));
        // Another allowed envelope realization disproves the stronger .125
        // bound that would result from pretending both neurons equal x.
        assert!(bound.lower <= actual(&p, &[x, x + 0.0625]));
    }
    assert_eq!(actual(&p, &[0.0, 0.0625]), 0.0625);
}

#[test]
fn all_other_head_terms_are_summed_before_root_box_minimum() {
    let p = SourceHeadProblem {
        root: vec![I { lo: -1.0, hi: 1.0 }],
        head: vec![env(1.0, 2.0, 2.0), env(0.0, 0.0, 0.0), env(1.0, 3.0, 3.0)],
        q: vec![I::p(1.0), I::p(0.0), I::p(-1.0)],
        bias: I::p(2.0),
    };
    let b = propose_and_check(&p, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    // Exact margin (x+2)-(x+3)+2=1. Independently minimizing the selected
    // group and unselected term gives1-4+2=-1; joint summation preserves x.
    assert!(b.lower > 0.99999999 && b.lower <= 1.0);
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        assert!(b.lower <= actual(&p, &[x + 2.0, 0.0, x + 3.0]));
    }
}

#[test]
fn four_neurons_include_every_closed_pattern_and_zero_boundary() {
    let p = SourceHeadProblem {
        root: vec![I { lo: -1.0, hi: 1.0 }],
        head: vec![
            env(1.0, 0.0, 0.0),
            env(-1.0, 0.0, 0.0),
            env(1.0, 0.25, 0.25),
            env(-1.0, -0.25, -0.25),
        ],
        q: vec![I::p(1.0), I::p(-1.0), I::p(0.5), I::p(-0.25)],
        bias: I::p(0.125),
    };
    let b = propose_and_check(&p, &[0, 1, 2, 3], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    assert_eq!(b.phases, 16);
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        assert!(b.lower <= actual(&p, &[x, -x, x + 0.25, -x - 0.25]));
    }
    let proposals: Vec<_> = (0..16)
        .map(|mask| Proposal {
            mask,
            phase: vec![0.0; 4],
        })
        .collect();
    assert!(check_complete(&p, &[0, 1, 2, 3], &proposals[..15], deadline()).is_none());
    let mut forged = proposals;
    forged[15].mask = 0;
    assert!(check_complete(&p, &[0, 1, 2, 3], &forged, deadline()).is_none());
}

#[test]
fn mutations_are_replayed_against_the_new_source_and_cannot_rebase_old_bound() {
    let old = twins();
    let oldb = propose_and_check(&old, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    let mut changed = old.clone();
    changed.head[1] = env(1.0, 0.25, 0.25);
    let newb = propose_and_check(&changed, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    assert!(oldb.applies_to(&old));
    assert!(!oldb.applies_to(&changed));
    assert_eq!(actual(&changed, &[0.0, 0.25]), -0.125);
    assert!(newb.lower <= -0.125);
    assert!(oldb.lower > 0.0);
    let mut wider = old.clone();
    wider.root[0] = I { lo: -2.0, hi: 2.0 };
    assert!(!oldb.applies_to(&wider));
    // max(old,new) is permitted ONLY when both bounds belong to this same
    // source/domain. A valid but unoptimized same-source certificate is kept
    // when better; a bound from the old source cannot be rebased here.
    assert!(newb.retained_with(&oldb).is_none());
    let zero: Vec<_> = (0..4)
        .map(|mask| Proposal {
            mask,
            phase: vec![0.0; 2],
        })
        .collect();
    let other = check_complete(&changed, &[0, 1], &zero, deadline()).unwrap();
    let retained = newb.retained_with(&other).unwrap();
    assert!(retained >= other.lower && retained >= newb.lower && retained <= -0.125);
}

#[test]
fn malformed_duals_nonfinite_shape_cap_and_deadline_fail_closed() {
    let p = twins();
    let mut candidates: Vec<_> = (0..4)
        .map(|mask| Proposal {
            mask,
            phase: vec![0.0; 2],
        })
        .collect();
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        candidates[0].phase[0] = value;
        assert!(check_complete(&p, &[0, 1], &candidates, deadline()).is_none());
    }
    assert!(propose_and_check(&p, &[0, 0], deadline(), MAX_PROPOSAL_ROUNDS).is_none());
    assert!(propose_and_check(&p, &[0, 2], deadline(), MAX_PROPOSAL_ROUNDS).is_none());
    assert!(propose_and_check(&p, &[0], deadline(), MAX_PROPOSAL_ROUNDS).is_none());
    assert!(propose_and_check(
        &p,
        &[0, 1],
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        MAX_PROPOSAL_ROUNDS
    )
    .is_none());
    let mut malformed = p.clone();
    malformed.head[0].lower.a.clear();
    assert!(propose_and_check(&malformed, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).is_none());
    let mut large = p;
    large.root = vec![I::p(0.0); MAX_INPUTS + 1];
    assert!(propose_and_check(&large, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).is_none());
}

#[test]
fn coefficient_uncertainty_fold_error_and_lost_unit_are_priced() {
    let mut p = twins();
    p.q[0] = I {
        lo: 1.0 - 1.0 / 1024.0,
        hi: 1.0 + 1.0 / 1024.0,
    };
    p.head[0] = env(1.0, -1.0 / 512.0, 1.0 / 512.0);
    let b = propose_and_check(&p, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    assert!(b.lower < 0.125);
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        for shift in [-1.0 / 512.0, 1.0 / 512.0] {
            for q in [p.q[0].lo, p.q[0].hi] {
                let value = p.bias.lo + q * (x + shift).max(0.0) - x.max(0.0);
                assert!(b.lower <= value);
            }
        }
    }
    let mut a = I::p(9_007_199_254_740_992.0);
    a = a
        .add(I::p(1.0))
        .unwrap()
        .add(I::p(-9_007_199_254_740_992.0))
        .unwrap();
    assert!(a.lo <= 1.0 && a.hi >= 1.0);
    let tiny = f64::from_bits(1);
    let dust = I::p(tiny).mul(I::p(0.5)).unwrap();
    assert!(dust.lo <= 0.0 && dust.hi >= tiny);
}

use super::super::bab::build_pack;
use super::super::head_group_source::HeadGroupSourceGuard;
use super::super::net::TwinNet;
use super::super::root::RootGates;
use super::super::spec::{TwinOpSpec, TwinSpec};

// Actual source adapter fixture: a positive trunk Conv/Relu followed by the
// authored two-Gemm classifier head. Both unperturbed head inputs equal x,
// yet separate unstable triangles lose their signed output cancellation.
pub(super) fn classifier_spec(
    outputs: usize,
    target: usize,
    second_shift: f64,
    rel: f64,
    berr: f64,
) -> TwinSpec {
    let mut weight = vec![0.0; outputs * 2];
    weight[target * 2] = 1.0;
    weight[target * 2 + 1] = -1.0;
    let mut bias = vec![0.0; outputs];
    bias[target] = 0.125;
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![2.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0; 2],
                bias: vec![-2.0, -2.0 + second_shift],
                bias_err: vec![berr; 2],
                weight_rel_err: rel,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight,
                bias,
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (outputs, 2),
            },
        ],
    }
}

fn nonlinear_twin_spec() -> TwinSpec {
    let mut spec = classifier_spec(2, 0, 0.0, 0.0, 0.0);
    if let TwinOpSpec::Conv { bias, .. } = &mut spec.ops[0] {
        bias[0] = 0.0;
    }
    if let TwinOpSpec::Gemm { bias, .. } = &mut spec.ops[3] {
        *bias = vec![-0.5; 2];
    }
    spec
}

#[test]
fn joint_native_source_pass_closes_nonlinear_twins_that_owned_envelopes_cannot() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&nonlinear_twin_spec()).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let affine = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    assert!(affine.lower < 0.0);
    let result = improve_root_margins(&eng, &pack, 0, &[1], &[-2.0], policy(), deadline()).unwrap();
    assert_eq!(result.phases_checked, 4);
    assert!(result.native_passes > 0 && result.native_columns > 0);
    assert!(result.native_improved_phases > 0);
    assert!(result.bounds[0] > 0.124999 && result.bounds[0] <= 0.125);
    // Analytic oracle: both source heads are exactly ReLU(x)-1/2, so
    // their final ReLUs cancel for every real x; min margin is exactly1/8.
    for x in [-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
        let y = x.max(0.0) - 0.5;
        assert_eq!(actual(&source, &[y, y]), 0.125);
        assert!(result.bounds[0] <= actual(&source, &[y, y]));
    }
}

#[test]
fn joint_native_nonlinear_closure_reaches_the_real_instance_token() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let spec = nonlinear_twin_spec();
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let config = |armed: bool| BabConfig {
        max_expansions: 0,
        deadline: Some(deadline()),
        head_groups: armed.then_some(policy()),
        ..BabConfig::default()
    };
    assert!(matches!(
        MarginRowBab::run(&net, &root, 0, &[1], config(false)),
        MarginRowOutcome::Unknown { .. }
    ));
    // Both authored heads equal ReLU(x)-1/2, so the exact margin is 1/8
    // for every real x. Complete native phase closure must reach the actual
    // frame/ledger authority without spending a tree expansion.
    match MarginRowBab::run(&net, &root, 0, &[1], config(true)) {
        MarginRowOutcome::Unsat(stats, _token) => assert_eq!(stats.root_closed_classes, 1),
        MarginRowOutcome::Unknown { reason, .. } => {
            panic!("joint nonlinear closure did not authorize: {reason}")
        }
    }
    let mut tied = spec;
    if let TwinOpSpec::Gemm { bias, .. } = &mut tied.ops[5] {
        bias[0] = 0.0;
    }
    let tied_net = TwinNet::compile(&tied).unwrap();
    let tied_root = RootGates::build(&tied_net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    assert!(matches!(
        MarginRowBab::run(&tied_net, &tied_root, 0, &[1], config(true)),
        MarginRowOutcome::Unknown { .. }
    ));
}

#[test]
fn completed_native_mask_survives_expiry_without_omitting_the_other_masks() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&nonlinear_twin_spec()).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let mut cover = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    assert!(cover.phase_lower[2] < 0.0 && cover.phase_lower[3] < 0.0);
    let phase2 = native::replay_one_for_test(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &Proposal {
            mask: 2,
            phase: vec![1.0, 0.0],
        },
        deadline(),
    )
    .unwrap();
    assert!(phase2 > 0.124999 && phase2 <= 0.125);
    cover.phase_lower[2] = cover.phase_lower[2].max(phase2);
    cover.lower = cover
        .phase_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    // The still-certified but weak mask3 prevents a positive class bound.
    assert!(cover.lower < 0.0);
    let before = cover.phase_lower.clone();
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let no_work =
        native::refine_complete_cover(&eng, &pack, &source, &[0, 1], &mut cover, expired).unwrap();
    assert_eq!(no_work.passes, 0);
    assert_eq!(cover.phase_lower, before);
    assert!(cover.phase_lower[2] > 0.0 && cover.lower < 0.0);
    let phase3 = native::replay_one_for_test(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &Proposal {
            mask: 3,
            phase: vec![0.0, 0.0],
        },
        deadline(),
    )
    .unwrap();
    cover.phase_lower[3] = cover.phase_lower[3].max(phase3);
    cover.lower = cover
        .phase_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert_eq!(cover.phases, 4);
    assert!(cover.lower > 0.124999 && cover.lower <= 0.125);
    // Numeric per-mask bounds cannot be rebased onto another group order.
    assert!(
        native::refine_complete_cover(&eng, &pack, &source, &[1, 0], &mut cover, deadline())
            .is_none()
    );
    let other = source.clone();
    assert!(
        native::refine_complete_cover(&eng, &pack, &other, &[0, 1], &mut cover, deadline())
            .is_none()
    );
}

#[test]
fn native_interval_seed_repair_rejects_the_lost_authored_unit() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let big = 9_007_199_254_740_992.0;
    let mut spec = classifier_spec(2, 0, 0.0, 0.0, 0.0);
    if let TwinOpSpec::Gemm { weight, .. } = &mut spec.ops[5] {
        *weight = vec![-big, big, 1.0, 0.0];
    }
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let lower = native::replay_one_for_test(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &Proposal {
            mask: 3,
            phase: vec![0.0, 0.0],
        },
        deadline(),
    )
    .unwrap();
    // Nominal seeds cancel, but the exact source margin at x=1 is -7/8.
    assert!(lower <= -0.875);
}

#[test]
fn native_signed_subnormal_products_are_charged_before_large_amplification() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let dust = f64::from_bits(1);
    let amplifier = 2.0_f64.powi(256);
    let amplified_dust = dust * amplifier;
    // All dyadic values below are exact binary64. At x=1 the authored
    // phase1111 margin is -3/8 * eta * 2^256. Four signed Gemm products
    // eta*(-3/8) round to zero, losing 3*eta/2 BEFORE the affine amplifier.
    // A single next_up(0) error floor cannot enclose that lost amount.
    let spec = TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![amplifier],
                shift: vec![0.0],
                scale_rel_err: 0.0,
                shift_err: vec![0.0],
                shape: (1, 1, 1),
            },
            TwinOpSpec::Flatten { input: 3 },
            TwinOpSpec::Gemm {
                input: 4,
                weight: vec![dust; 4],
                bias: vec![-amplified_dust / 2.0; 4],
                bias_err: vec![0.0; 4],
                weight_rel_err: 0.0,
                shape: (4, 1),
            },
            TwinOpSpec::Relu { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![
                    -3.0 / 8.0,
                    -3.0 / 8.0,
                    -3.0 / 8.0,
                    -3.0 / 8.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                bias: vec![3.0 * amplified_dust / 8.0, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 4),
            },
        ],
    };
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[0.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let at_one = -3.0 * amplified_dust / 8.0;
    assert!(at_one < 0.0 && at_one.is_normal());
    assert_eq!(actual(&source, &[amplified_dust / 2.0; 4]), at_one);
    let lower = native::replay_one_for_test(
        &eng,
        &pack,
        &source,
        &[0, 1, 2, 3],
        &Proposal {
            mask: 15,
            phase: vec![0.0; 4],
        },
        deadline(),
    )
    .expect("absolute underflow budget should support this finite source");
    assert!(
        lower <= at_one,
        "native lower {lower:e} exceeded exact source {at_one:e}"
    );
}

#[test]
fn source_capture_refuses_unrepaired_envelope_policies_at_every_width() {
    use super::super::errlane::ErrLane;
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&nonlinear_twin_spec()).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let ordinary = BackwardEngine::new(&net, &root);
    let pack = build_pack(&ordinary, None, &[]).unwrap();
    assert!(pack.owned_for_head_groups(&ordinary));
    for lane in [ErrLane::Envelope, ErrLane::PerChannel] {
        // Even a narrow envelope-configured pass that publishes PerCol is
        // refused; the original complete source pack may be wider.
        let changed = BackwardEngine::new(&net, &root).with_err_lane(lane);
        assert!(!pack.owned_for_head_groups(&changed));
        let other = build_pack(&changed, None, &[]).unwrap();
        assert!(!other.owned_for_head_groups(&changed));
        assert!(from_owned_pack(&changed, &other, deadline()).is_none());
    }
}

#[test]
fn source_generation_refuses_patch_roots_and_patch_refresh_gates() {
    use super::super::{BackwardIntermRuntimePolicy, RootBoxSource, RootPrecisionRuntimePolicy};
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&nonlinear_twin_spec()).unwrap();
    let patch_precision = RootPrecisionRuntimePolicy {
        box_source: RootBoxSource::PatchBackward,
        ..RootPrecisionRuntimePolicy::default()
    };
    let patch_root = RootGates::build_with_policy_for_test(
        &net,
        &[-1.0],
        &[1.0],
        RoundMode::Outward,
        patch_precision,
        None,
    )
    .unwrap()
    .0;
    assert!(patch_root.head_group_source.is_none());

    let mut patch_bi = BackwardIntermRuntimePolicy {
        enabled: true,
        secs: 0.0,
        ..BackwardIntermRuntimePolicy::default()
    };
    patch_bi.patch.enabled = true;
    let bi_root = RootGates::build_retaining_with_policy(
        &net,
        &[-1.0],
        &[1.0],
        RoundMode::Outward,
        None,
        None,
        &[],
        RootPrecisionRuntimePolicy::default(),
        patch_bi,
        None,
    )
    .unwrap()
    .0;
    assert!(bi_root.head_group_source.is_none());

    let root =
        RootGates::build_prec(&net, &[-1.0], &[1.0], RoundMode::Outward, None, false).unwrap();
    assert!(root.head_group_source.is_some());
    let past = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    let (dense, _) = super::super::backward_interm::refresh_root_intermediates_with_patch(
        &net, &root, 1, 1, false, None, past, None,
    )
    .unwrap();
    assert!(dense.head_group_source.is_some());
    let (patch, _) = super::super::backward_interm::refresh_root_intermediates_with_patch(
        &net,
        &root,
        1,
        1,
        false,
        Some(super::super::patch::PatchCfg::new(0.5, 256)),
        past,
        None,
    )
    .unwrap();
    assert!(patch.head_group_source.is_none());
}

#[test]
fn source_chord_qualification_rejects_underflowed_identity_on_an_unstable_box() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&nonlinear_twin_spec()).unwrap();
    let mut root =
        RootGates::build_prec(&net, &[-1.0], &[1.0], RoundMode::Outward, None, false).unwrap();
    assert!(root_lines_qualified(&root));
    let dust = f64::from_bits(1);
    let (l, u) = (vec![-dust], vec![0.25]);
    let (a, mut s, mut c) = super::super::root::gates_from_box(&l, &u);
    assert_eq!((s[0], c[0]), (1.0, 0.0));
    let gate = &mut root.layers[0];
    gate.l = l.clone();
    gate.u = u.clone();
    gate.alpha = a;
    gate.s = s.clone();
    gate.c = c.clone();
    gate.ms = vec![1.0];
    assert!(!root_lines_qualified(&root));
    assert!(
        super::super::head_group_source::YPackSource::capture(&BackwardEngine::new(&net, &root))
            .is_none()
    );
    // The shortcut must inspect the source box: s=1,c=0 is valid only if
    // this box is active. The actual unstable chord receives a repair.
    super::super::root::repair_upper_lines(&l, &u, &mut s, &mut c);
    root.layers[0].s = s;
    root.layers[0].c = c;
    assert!(root_lines_qualified(&root));
}

#[test]
fn directed_source_reconstruction_accepts_existing_tableau_precision_modes() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&nonlinear_twin_spec()).unwrap();
    for (compute, store) in [(false, false), (true, false), (true, true)] {
        let root = RootGates::build_prec_store(
            &net,
            &[-1.0],
            &[1.0],
            RoundMode::Outward,
            None,
            compute,
            store,
        )
        .unwrap();
        assert!(root_lines_qualified(&root));
        let eng = BackwardEngine::new(&net, &root);
        let pack = build_pack(&eng, None, &[]).unwrap();
        let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
        bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
        for x in [-1.0_f64, 0.0, 0.5, 1.0] {
            let y = x.max(0.0) - 0.5;
            assert!(source.head.iter().all(|e| e.y.lo <= y && y <= e.y.hi));
        }
    }
}

fn policy() -> HeadGroupPolicy {
    HeadGroupPolicy::new(5.0, 2, 4).unwrap()
}

#[test]
fn adapter_uses_owned_source_rows_and_preserves_reordered_class_identity_and_strict_ties() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let mut spec = classifier_spec(10, 3, 0.0, 0.0, 0.0);
    // Class1 exactly ties the target, while all other rows have the positive
    // constant margin. It must remain open under the lane's strict >0 rule.
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[5] {
        weight[2] = 1.0;
        weight[3] = -1.0;
        bias[1] = 0.125;
    }
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let old = [-0.5, 0.01, 0.0];
    let result =
        improve_root_margins(&eng, &pack, 3, &[8, 2, 1], &old, policy(), deadline()).unwrap();
    assert_eq!(result.groups_checked, 1);
    assert_eq!(result.phases_checked, 4);
    assert_eq!(result.classes_improved, 1);
    assert_eq!(result.candidates[0].adversary, 8);
    assert!(result.bounds[0] > 0.124999 && result.bounds[0] <= 0.125);
    assert_eq!(result.bounds[1].to_bits(), old[1].to_bits());
    assert_eq!(result.bounds[2].to_bits(), 0.0f64.to_bits());
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        let actual = 0.125 + x.max(0.0) - x.max(0.0);
        assert!(result.bounds[0] <= actual);
    }
}

#[test]
fn adapter_mutated_unsafe_source_cannot_reuse_old_pack_or_close_the_new_margin() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let old = TwinNet::compile(&classifier_spec(2, 0, 0.0, 0.0, 0.0)).unwrap();
    let changed = TwinNet::compile(&classifier_spec(2, 0, 0.25, 0.0, 0.0)).unwrap();
    let old_root = RootGates::build(&old, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let changed_root =
        RootGates::build(&changed, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let old_eng = BackwardEngine::new(&old, &old_root);
    let changed_eng = BackwardEngine::new(&changed, &changed_root);
    let old_pack = build_pack(&old_eng, None, &[]).unwrap();
    assert!(improve_root_margins(
        &changed_eng,
        &old_pack,
        0,
        &[1],
        &[-0.5],
        policy(),
        deadline()
    )
    .is_none());
    let changed_pack = build_pack(&changed_eng, None, &[]).unwrap();
    let result = improve_root_margins(
        &changed_eng,
        &changed_pack,
        0,
        &[1],
        &[-0.5],
        policy(),
        deadline(),
    )
    .unwrap();
    // Authored exact source at x=0 has margin .125-.25=-.125.
    assert!(result.bounds[0] <= -0.125);
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        assert!(result.bounds[0] <= 0.125 + x.max(0.0) - (x + 0.25).max(0.0));
    }
}

#[test]
fn adapter_different_root_and_parity_pass_refuse_source_authority() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&classifier_spec(2, 0, 0.0, 0.0, 0.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let other_root = RootGates::build(&net, &[-0.5], &[0.5], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let other_eng = BackwardEngine::new(&net, &other_root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    assert!(
        improve_root_margins(&other_eng, &pack, 0, &[1], &[-0.5], policy(), deadline()).is_none()
    );
    let parity_root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Parity, None).unwrap();
    let parity_eng = BackwardEngine::new(&net, &parity_root);
    let parity_pack = build_pack(&parity_eng, None, &[]).unwrap();
    assert!(improve_root_margins(
        &parity_eng,
        &parity_pack,
        0,
        &[1],
        &[-0.5],
        policy(),
        deadline()
    )
    .is_none());
}

#[test]
fn adapter_requires_complete_directed_error_arrays_and_refuses_malformed_rows() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&classifier_spec(2, 0, 0.0, 0.0, 0.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    for case in 0..6 {
        let mut pack = build_pack(&eng, None, &[]).unwrap();
        match case {
            0 => pack.al.e = None,
            1 => pack.al.e.as_mut().unwrap()[[0, 0]] = -1.0,
            2 => pack.au.e.as_mut().unwrap()[[0, 1]] = f64::NAN,
            3 => pack.au.eb[0] = -1.0,
            4 => pack.al.e = Some(ndarray::Array2::zeros((2, 1))),
            5 => pack.al.b[0] = f64::INFINITY,
            _ => unreachable!(),
        }
        assert!(
            improve_root_margins(&eng, &pack, 0, &[1], &[-0.5], policy(), deadline()).is_none(),
            "case {case}"
        );
    }
}

#[test]
fn adapter_charges_folded_head_error_ball_instead_of_treating_nominal_y_as_exact() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let rel = 1.0 / 64.0;
    let berr = 1.0 / 32.0;
    let net = TwinNet::compile(&classifier_spec(2, 0, 0.0, rel, berr)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let result = improve_root_margins(&eng, &pack, 0, &[1], &[-1.0], policy(), deadline()).unwrap();
    assert_eq!(result.groups_checked, 1);
    // Every extreme realization of the authored fold ball is covered. All
    // numbers on this grid are dyadic; no sampling tolerance is necessary.
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        let low = (x + 2.0) * (1.0 - rel) - 2.0 - berr;
        let high = (x + 2.0) * (1.0 + rel) - 2.0 + berr;
        let worst = 0.125 + low.max(0.0) - high.max(0.0);
        assert!(
            result.bounds[0] <= worst,
            "x={x} lower={} actual={worst}",
            result.bounds[0]
        );
    }
    // At x=1 this ball has a real -.03125 margin; dropping its fold charge
    // would preserve the nominal +.125 cancellation and falsely close it.
    assert!(result.bounds[0] <= -0.03125);
}

#[test]
fn adapter_authored_final_subtraction_prices_a_lost_unit_before_phase_replay() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let big = 9_007_199_254_740_992.0;
    let mut spec = classifier_spec(2, 0, 0.0, 0.0, 0.0);
    if let TwinOpSpec::Gemm { weight, .. } = &mut spec.ops[5] {
        *weight = vec![-big, big, 1.0, 0.0];
    }
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let canceled = source.q[0].add(source.q[1]).unwrap();
    assert!(canceled.lo <= -1.0 && canceled.hi >= -1.0);
    assert_eq!((-big - 1.0) + big, 0.0);
    let result = improve_root_margins(&eng, &pack, 0, &[1], &[-2.0], policy(), deadline()).unwrap();
    // The exact source margin is .125-ReLU(x), and violates at x=1.
    // Nominal q subtraction loses that negative ReLU term, leaving a false
    // positive constant. Source replay must not close the authored row.
    assert!(result.bounds[0] <= -0.875);
}

#[test]
fn adapter_typed_caps_row_identity_and_expiry_fail_closed() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    for args in [
        (0.0, 1, 1),
        (6.0, 1, 1),
        (f64::NAN, 1, 1),
        (1.0, 0, 1),
        (1.0, 3, 1),
        (1.0, 1, 0),
        (1.0, 1, 5),
    ] {
        assert!(HeadGroupPolicy::new(args.0, args.1, args.2).is_none());
    }
    let p = policy();
    assert_eq!(p.max_seconds(), 5.0);
    assert_eq!(p.groups_per_margin(), 2);
    assert_eq!(p.proposal_rounds(), 4);
    let net = TwinNet::compile(&classifier_spec(2, 0, 0.0, 0.0, 0.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(improve_root_margins(&eng, &pack, 0, &[1], &[-0.5], p, expired).is_none());
    for (target, adversaries, old) in [
        (0, vec![0], vec![-0.5]),
        (0, vec![1, 1], vec![-0.5; 2]),
        (2, vec![1], vec![-0.5]),
        (0, vec![1], vec![]),
    ] {
        assert!(
            improve_root_margins(&eng, &pack, target, &adversaries, &old, p, deadline()).is_none()
        );
    }
}

use super::super::bab::{
    refine_head_groups, root_eval_reuse, root_eval_store, BabConfig, EpochFrameRoute, MarginRowBab,
    MarginRowOutcome,
};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn final_root_hook_keeps_off_bits_and_raw_memo_while_armed_hits_still_refine() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&classifier_spec(2, 0, 0.0, 0.0, 0.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let raw = root_eval_store(&eng, &net, 0, &[1]).unwrap();
    assert!(
        raw.dj[0] <= 0.0,
        "fixture must remain open under the independent-head root"
    );
    let raw_bits = bits(&raw.dj);
    let mut hit = root_eval_reuse(&eng, &net, 0, &[1]).unwrap();
    assert_eq!(bits(&hit.dj), raw_bits);
    let off = BabConfig {
        deadline: Some(deadline()),
        ..BabConfig::default()
    };
    refine_head_groups(&eng, 0, &[1], &mut hit, &off);
    assert_eq!(bits(&hit.dj), raw_bits);
    assert_eq!(hit.tree_classes, raw.tree_classes);

    let on = BabConfig {
        head_groups: Some(policy()),
        ..off
    };
    refine_head_groups(&eng, 0, &[1], &mut hit, &on);
    assert!(hit.dj[0] > 0.0 && hit.dj[0] <= 0.125);
    assert_eq!((hit.tree_classes).len(), 0);
    let refined_bits = bits(&hit.dj);

    // A subsequent raw memo publication must still contain only the old
    // evaluator's result. Its consumer independently runs the optional family.
    let raw_again = root_eval_store(&eng, &net, 0, &[1]).unwrap();
    assert_eq!(bits(&raw_again.dj), raw_bits);
    let mut second_hit = root_eval_reuse(&eng, &net, 0, &[1]).unwrap();
    assert_eq!(bits(&second_hit.dj), raw_bits);
    refine_head_groups(&eng, 0, &[1], &mut second_hit, &on);
    assert_eq!(bits(&second_hit.dj), refined_bits);
}

#[test]
fn final_root_hook_refuses_nested_inherited_head_ybox_and_expired_original_deadline() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&classifier_spec(2, 0, 0.0, 0.0, 0.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    for case in 0..5 {
        let mut re = root_eval_store(&eng, &net, 0, &[1]).unwrap();
        let old_bounds = bits(&re.dj);
        let old_classes = re.tree_classes.clone();
        let mut cfg = BabConfig {
            deadline: Some(deadline()),
            head_groups: Some(policy()),
            ..BabConfig::default()
        };
        match case {
            0 => cfg.epoch_frame_route = EpochFrameRoute::NestedFull,
            1 => cfg.epoch_frame_route = EpochFrameRoute::NestedIncremental,
            2 => cfg.initial_heads = vec![(0, 1)],
            3 => cfg.initial_ybox = Some((vec![0.0; 2], vec![1.0; 2])),
            4 => cfg.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
            _ => unreachable!(),
        }
        refine_head_groups(&eng, 0, &[1], &mut re, &cfg);
        assert_eq!(bits(&re.dj), old_bounds, "case {case}");
        assert_eq!(re.tree_classes, old_classes, "case {case}");
    }
}

#[test]
fn all_nine_source_margins_need_strict_closure_before_actual_instance_token() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    let spec = classifier_spec(10, 3, 0.0, 0.0, 0.0);
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let adversaries: Vec<_> = (0..10).filter(|i| *i != 3).collect();
    let off = BabConfig {
        max_expansions: 0,
        deadline: Some(deadline()),
        ..BabConfig::default()
    };
    assert!(matches!(
        MarginRowBab::run(&net, &root, 3, &adversaries, off),
        MarginRowOutcome::Unknown { .. }
    ));
    let on = BabConfig {
        max_expansions: 0,
        deadline: Some(deadline()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    // This is the real driver, cover authorization and UnsatToken path; the
    // optional numeric report cannot construct this token by itself.
    match MarginRowBab::run(&net, &root, 3, &adversaries, on) {
        MarginRowOutcome::Unsat(stats, _token) => assert_eq!(stats.root_closed_classes, 9),
        MarginRowOutcome::Unknown { reason, .. } => {
            panic!("safe exact source fixture did not authorize: {reason}")
        }
    }

    let mut tied = spec;
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut tied.ops[5] {
        weight[2] = 1.0;
        weight[3] = -1.0;
        bias[1] = 0.125;
    }
    let tied_net = TwinNet::compile(&tied).unwrap();
    let tied_root = RootGates::build(&tied_net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let tied_cfg = BabConfig {
        max_expansions: 0,
        deadline: Some(deadline()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    // Eight positive margins and one exact zero never authorize all classes.
    assert!(matches!(
        MarginRowBab::run(&tied_net, &tied_root, 3, &adversaries, tied_cfg),
        MarginRowOutcome::Unknown { .. }
    ));
}

// Continuous cancellation advice is independently replayed below. These
// tests exercise native proposals; no AY-provider integration is claimed.

fn continuous_advice_fixture() -> SourceHeadProblem {
    SourceHeadProblem {
        root: vec![I { lo: -1.0, hi: 1.0 }],
        head: vec![env(1.0, 0.0, 0.0), env(1.0, 0.5, 0.5)],
        q: vec![I::p(1.0), I::p(-3.0 / 8.0)],
        bias: I::p(13.0 / 64.0),
    }
}

fn continuous_advice_complete_proposals() -> Vec<Proposal> {
    vec![
        Proposal {
            mask: 0,
            phase: vec![0.0, 0.0],
        },
        Proposal {
            mask: 1,
            phase: vec![1.0, 0.0],
        },
        Proposal {
            mask: 2,
            phase: vec![3.0 / 8.0, 0.0],
        },
        Proposal {
            mask: 3,
            phase: vec![5.0 / 8.0, 0.0],
        },
    ]
}

#[test]
fn continuous_advice_closes_a_margin_that_every_native_grid_pair_misses() {
    // Exact integer oracle for all 49 pairs, not a floating LP optimum.
    // A=a/32; B=b/32; each result is the exact lower bound multiplied by64.
    let a_units = [0_i64, 8, 16, 32, 64, 128, 256];
    let b_units = [0_i64, 3, 6, 12, 24, 48, 96];
    let phase11_max: Vec<_> = a_units
        .iter()
        .map(|&a| {
            b_units
                .iter()
                .map(|&b| 13 - (12 + b) - 2 * (20 - a - b).abs())
                .max()
                .unwrap()
        })
        .collect();
    let phase01_max: Vec<_> = a_units
        .iter()
        .map(|&a| {
            b_units
                .iter()
                .map(|&b| 13 - (12 + b) - 2 * (a - 12 - b).abs())
                .max()
                .unwrap()
        })
        .collect();
    assert_eq!(phase11_max, vec![-27, -11, -4, -23, -87, -215, -471]);
    assert_eq!(phase01_max, vec![-23, -7, -4, -27, -55, -135, -391]);
    assert_eq!(*phase11_max.iter().max().unwrap(), -4);
    assert_eq!(*phase01_max.iter().max().unwrap(), -4);

    let source = continuous_advice_fixture();
    let grid = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    assert_eq!(grid.phases, 4);
    assert!(grid.lower <= -1.0 / 16.0);
    let proposals = continuous_advice_complete_proposals();
    let replay = check_complete(&source, &[0, 1], &proposals, deadline()).unwrap();
    assert_eq!(replay.phases, 4);
    assert!(replay.lower > 1.0 / 128.0 && replay.lower <= 1.0 / 64.0);

    // Exact source minimum at the shared closed-phase boundary x=0.
    assert_eq!(actual(&source, &[0.0, 0.5]), 1.0 / 64.0);
    assert!(check_complete(&source, &[0, 1], &proposals[..3], deadline()).is_none());
}

#[test]
fn continuous_advice_is_replayed_through_the_real_producer_owned_source() {
    let _source_scope = HeadGroupSourceGuard::enter(true);
    // The existing fixture supplies Conv(x+2), stable ReLU, Flatten,
    // Gemm(z-2,z-3/2), head ReLU, and an unfolded two-class final Gemm.
    let mut spec = classifier_spec(2, 0, 0.5, 0.0, 0.0);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[5] {
        weight[1] = -3.0 / 8.0;
        bias[0] = 13.0 / 64.0;
    } else {
        panic!("fixture final op must be Gemm");
    }
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();

    let grid = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    assert_eq!(grid.phases, 4);
    assert!(grid.lower < 0.0);
    let replay = check_complete(
        &source,
        &[0, 1],
        &continuous_advice_complete_proposals(),
        deadline(),
    )
    .unwrap();
    assert_eq!(replay.phases, 4);
    assert!(replay.lower > 1.0 / 128.0 && replay.lower <= 1.0 / 64.0);
    // The combined-Gemm ranker may also propose its continuous coordinate
    // roots. This actual driver check does not claim an AY provider was used.
    let result = improve_root_margins(&eng, &pack, 0, &[1], &[-1.0], policy(), deadline()).unwrap();
    assert!(result.native_improved_phases >= 2);
    assert!(result.bounds[0] > 1.0 / 128.0 && result.bounds[0] <= 1.0 / 64.0);
    // No token or AY optimum is used: the owned source numeric replay alone
    // accepts the new lower bound. A future test must exercise the AY provider.
}
