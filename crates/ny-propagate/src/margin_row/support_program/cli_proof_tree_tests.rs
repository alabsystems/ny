// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Complete real conditional enumeration reaches G2 and the existing tree
//! authorizer. No control supplies a scalar, token, or synthetic certificate.

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::certify::{GuardId, PaidCoverError, PaidCoverLedger, TokenScope, UnsatSource};
use crate::margin_row::bab::{
    MarginRowBab, PreparedTreeCertificate, PreparedTreeOptions, PreparedTreeOutcome,
    PreparedTreeSchedule,
};
use crate::margin_row::spec::TwinOpSpec;
use crate::margin_row::support_program::{PreparedDomain, PreparedPhaseSide};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 32 << 20;
fn clock() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(clock()),
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(CAP).unwrap();
    (budget, memory, cancel)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn fixture(classes: usize, target: usize, fold: bool, separated: bool, margin: f64) -> TwinSpec {
    let mut weights = vec![0.0; 2 * classes];
    weights[2 * target] = 1.0;
    weights[2 * target + 1] = -1.0;
    let mut bias = vec![0.0; classes];
    bias[target] = margin;
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![if fold { 1.0 / 1024.0 } else { 0.0 }],
                weight_rel_err: if fold { 1.0 / 512.0 } else { 0.0 },
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
                weight: vec![1.0, 1.0],
                bias: vec![-1.0, if separated { -2.0 } else { -1.0 }],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: weights,
                bias,
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (classes, 2),
            },
        ],
    }
}
fn session(
    spec: TwinSpec,
    budget: &WorkBudget,
    options: PreparedRootProofOptions,
) -> PreparedRootSession {
    PreparedRootSession::compile_declared_twin(
        spec,
        &[0.0],
        &[3.0],
        RootPrecisionRuntimePolicy::default(),
        options,
        budget,
        Some(clock()),
    )
    .unwrap()
}
fn original_open(session: &PreparedRootSession, target: usize) -> PreparedRootOpen {
    let order: Vec<_> = (0..session.n_out())
        .filter(|&c| c != target)
        .rev()
        .collect();
    let PreparedRootOutcome::Open(open) = session.try_prove_root(target, &order, None).unwrap()
    else {
        panic!("real duplicate-head root must be open")
    };
    assert!(open.receipt().ordinary_completed);
    assert!(!open.receipt().terminal_closed);
    open
}
fn certificate(open: PreparedRootOpen) -> PreparedTreeCertificate {
    let options = PreparedTreeOptions {
        max_expansions: 2,
        lru_capacity: 4,
        schedule: PreparedTreeSchedule::Serial,
    };
    let PreparedTreeOutcome::Covered(cert) =
        MarginRowBab::run_prepared(open, options, None).unwrap()
    else {
        panic!("actual head children must close")
    };
    cert
}
fn options() -> PreparedRootProofOptions {
    PreparedRootProofOptions {
        prefer_patch: false,
        ..PreparedRootProofOptions::default()
    }
}
fn exact_original_margin(x: BigRational, separated: bool, margin: BigRational) -> BigRational {
    let trunk = x.max(BigRational::zero());
    let left = (trunk.clone() - BigRational::from_integer(1.into())).max(BigRational::zero());
    let right = (trunk - BigRational::from_integer(if separated { 2.into() } else { 1.into() }))
        .max(BigRational::zero());
    margin + left - right
}

#[test]
fn cli_tree_factory_real_open_root_paid_children_cover_g2_and_sole_tree_authorizer() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let open = original_open(&session, 0);
    let cert = certificate(open);
    assert_eq!(cert.receipt().bound_leaves, 2);
    assert!(cert.cover(None).unwrap().min_margin().is_finite());
    let proof = PreparedRootProof::from_prepared_tree(cert, None).unwrap();
    assert_eq!(proof.tree_receipt().unwrap().expansions, 1);
    assert_eq!(proof.tree_receipt().unwrap().bound_leaves, 2);
    assert!(proof.session().same_source(&session));
    assert!(proof.budget().shares_ledger(&budget));
    assert_eq!(proof.publication.token().source(), UnsatSource::Authorized);
    assert_eq!(proof.publication.token().scope(), TokenScope::Instance);
    assert!(proof.publication.receipt().containment_points > 0);
    assert!(proof.publication.receipt().containment_checks > 0);
    assert!(!proof.publication.receipt().containment_skipped);
    assert!(proof.min_margin().is_finite() && proof.min_margin() > 0.0);
    for x in [
        BigRational::zero(),
        BigRational::new(1.into(), 2.into()),
        BigRational::from_integer(1.into()),
        BigRational::from_integer(3.into()),
    ] {
        assert!(
            BigRational::from_float(proof.min_margin()).unwrap()
                <= exact_original_margin(x, false, BigRational::new(3.into(), 4.into()))
        );
    }
    proof.checkpoint(None).unwrap();
    drop(session);
    assert!(live(&memory) > 0);
    drop(proof);
    assert_eq!(live(&memory), 0);
}
#[test]
fn cli_tree_factory_nonzero_target_complete_original_order_above64_and_canonical_token_match() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(66, 17, false, false, 0.75), &budget, options());
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&session, 17)), None)
            .unwrap();
    let order: Vec<_> = (0..66).filter(|&c| c != 17).rev().collect();
    assert_eq!(proof.target(), 17);
    assert_eq!(proof.classes(), order);
    assert_eq!(proof.publication.token().classes().len(), 65);
    assert_eq!(proof.publication.token().classes()[64], 65);
    proof.validate_objective(17, &order, None).unwrap();
    let reversed: Vec<_> = order.iter().rev().copied().collect();
    assert!(proof.validate_objective(17, &reversed, None).is_err());
    assert!(proof.validate_objective(17, &order[..64], None).is_err());
    assert!(proof.validate_objective(0, &order, None).is_err());
}
#[test]
fn cli_tree_factory_duplicate_equal_model_hashes_do_not_replace_actual_source_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let first = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let other = session(fixture(2, 0, false, false, 0.75), &budget, options());
    assert_eq!(first.source_digest(), other.source_digest());
    assert_eq!(first.compiled_digest(), other.compiled_digest());
    assert!(!first.same_source(&other));
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&first, 0)), None).unwrap();
    assert!(proof.session().same_source(&first));
    assert!(!proof.session().same_source(&other));
    assert!(proof.publication.source_owner().same_source(first.source()));
    assert!(!proof.publication.source_owner().same_source(other.source()));
    // The factory derives its source from the certificate; it has no expected
    // session/hash/objective parameter with which to relabel this proof.
}
#[test]
fn cli_tree_factory_last_clone_retains_all_actual_leaf_and_source_claims_once() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&session, 0)), None)
            .unwrap();
    drop(session);
    let before = live(&memory);
    let before_receipt = budget.receipt().unwrap();
    let alias = proof.clone();
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, before_receipt.admitted);
    drop(proof);
    assert_eq!(live(&memory), before);
    alias.checkpoint(None).unwrap();
    assert_eq!(alias.tree_receipt().unwrap().bound_leaves, 2);
    drop(alias);
    assert_eq!(live(&memory), 0);
}
#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn cli_tree_factory_successful_earlier_factory_and_ratification_clocks_survive_aliases() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let first = Instant::now() + Duration::from_secs(30);
    let second = Instant::now() + Duration::from_secs(20);
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&session, 0)), Some(first))
            .unwrap();
    assert!(proof.deadline(None).unwrap() <= first);
    let earlier = proof
        .clone()
        .ratified(&GuardReport::new(), Some(second))
        .unwrap();
    assert!(earlier.deadline(Some(clock())).unwrap() <= second);
    assert!(earlier.session().deadline(None).unwrap() <= second);
    assert!(earlier.publication.deadline(None).unwrap() <= second);
    assert_eq!(proof.deadline(None).unwrap(), first);
    assert!(earlier.clone().deadline(None).unwrap() <= second);
}
#[test]
fn cli_tree_factory_cancelled_certificate_never_admits_new_publication_or_detaches_token() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let open = original_open(&session, 0);
    let cert = certificate(open.clone());
    let before = budget.receipt().unwrap();
    cancel.cancel();
    assert!(PreparedRootProof::from_prepared_tree(cert, None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert!(live(&memory) > 0);
    drop(open);
    drop(session);
    assert_eq!(live(&memory), 0);
}
#[test]
fn cli_tree_factory_expired_handoff_refuses_before_new_authority_admission() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let open = original_open(&session, 0);
    let cert = certificate(open.clone());
    let before = budget.receipt().unwrap();
    assert!(PreparedRootProof::from_prepared_tree(
        cert,
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    open.checkpoint(None).unwrap();
}
#[test]
fn cli_tree_factory_original_occupied_cap_preserves_real_source_without_fresh_budget() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let open = original_open(&session, 0);
    let cert = certificate(open.clone());
    let r = budget.receipt().unwrap();
    let remaining = CAP - r.live_scratch_bytes - r.live_persistent_bytes;
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: remaining - 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    lease.mark_materialized().unwrap();
    let held = lease.commit_persistent().unwrap();
    assert!(PreparedRootProof::from_prepared_tree(cert, None).is_err());
    drop(held);
    open.checkpoint(None).unwrap();
    assert!(live(&memory) > 0);
    drop(open);
    drop(session);
    assert_eq!(live(&memory), 0);
}
#[test]
fn cli_tree_factory_real_empty_subregion_cannot_supply_an_all_empty_original_partition() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(2, 0, false, true, 0.75), &budget, options());
    let open = original_open(&session, 0);
    let domain = PreparedDomain::from_open(open.clone(), None)
        .unwrap()
        .split_head(0, PreparedPhaseSide::Inactive, None)
        .unwrap()
        .split_head(1, PreparedPhaseSide::Active, None)
        .unwrap();
    let evaluation = crate::margin_row::bab::evaluate_prepared_domain(&domain, None).unwrap();
    let slack = evaluation
        .crossing()
        .expect("real t<=1 and t>=2 child is empty");
    assert!(slack.is_finite() && slack > 0.0);
    let path = domain.cover_path(None).unwrap();
    let mut ledger = PaidCoverLedger::new(&budget, 1, clock()).unwrap();
    ledger
        .leaf(
            Some(path.path(None).unwrap()),
            crate::certify::margin_row::empty_evidence(
                crate::margin_row::RoundMode::Outward,
                slack,
            ),
            None,
        )
        .unwrap();
    assert!(matches!(
        ledger.prove(open.classes(), None),
        Err(PaidCoverError::Cover(_))
    ));
    // The only actual complete tree still retains positive finite bound leaves.
    let proof = PreparedRootProof::from_prepared_tree(certificate(open), None).unwrap();
    assert!(proof.tree_receipt().unwrap().bound_leaves > 0);
    assert!(proof.min_margin().is_finite());
}
#[test]
fn cli_tree_factory_actual_original_shared_fold_uncertainty_keeps_global_rational_lower() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(2, 0, true, false, 0.75), &budget, options());
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&session, 0)), None)
            .unwrap();
    assert!(proof.min_margin() > 0.0 && proof.min_margin() <= 0.75);
    for coefficient in [
        BigRational::new(511.into(), 512.into()),
        BigRational::new(513.into(), 512.into()),
    ] {
        for bias in [
            BigRational::new((-1).into(), 1024.into()),
            BigRational::new(1.into(), 1024.into()),
        ] {
            for input in [BigRational::zero(), BigRational::from_integer(3.into())] {
                let exact = exact_original_margin(
                    coefficient.clone() * input + bias.clone(),
                    false,
                    BigRational::new(3.into(), 4.into()),
                );
                assert!(BigRational::from_float(proof.min_margin()).unwrap() <= exact);
            }
        }
    }
}
#[test]
fn cli_tree_factory_downgrade_guards_and_requested_precision_cannot_grant_new_authority() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&session, 0)), None)
            .unwrap();
    let mut guards = GuardReport::new();
    guards.trip(
        GuardId::SpecAgreement,
        "independent original objective disagreement",
    );
    assert!(proof.clone().ratified(&guards, None).is_err());
    let mut policy = proof.policy();
    policy.root_f32 = true;
    assert!(proof.publication.validate_policy(policy, None).is_err());
    policy.root_f32 = false;
    policy.tableau_f32 = true;
    assert!(proof.publication.validate_policy(policy, None).is_err());
    proof.checkpoint(None).unwrap();
}
#[test]
fn cli_tree_factory_actual_negative_subnormal_original_margin_stays_open_without_certificate() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let minimum = f64::from_bits(1);
    let session = session(fixture(2, 0, false, false, -minimum), &budget, options());
    let open = original_open(&session, 0);
    let result = MarginRowBab::run_prepared(
        open,
        PreparedTreeOptions {
            max_expansions: 2,
            lru_capacity: 4,
            schedule: PreparedTreeSchedule::Serial,
        },
        None,
    )
    .unwrap();
    let PreparedTreeOutcome::Open(open) = result else {
        panic!("actual original exact negative margin must never create a complete positive tree certificate")
    };
    assert!(open.open_domains() > 0);
    assert!(open.receipt().bound_leaves < 2);
    let exact = BigRational::from_float(-minimum).unwrap();
    assert!(exact < BigRational::zero());
}

#[test]
fn cli_tree_replay_actual_root_exports_original_topology_and_complete_unsplit_objective() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(3, 1, false, false, 4.0), &budget, options());
    let PreparedRootOutcome::Proven(proof) = session.try_prove_root(1, &[2, 0], None).unwrap()
    else {
        panic!("actual original constant margin must close at root")
    };
    let before = live(&memory);
    let stats = proof.replay_stats(None).unwrap();
    assert_eq!(
        (
            stats.site_count,
            stats.leaf_count,
            stats.literal_count,
            stats.max_depth
        ),
        (2, 1, 0, 0)
    );
    let mut sites = Vec::new();
    proof
        .visit_replay_sites(None, &mut |op, width| {
            sites.push((op, width));
            Ok(())
        })
        .unwrap();
    assert_eq!(sites, [(1, 1), (4, 2)]);
    let mut visited = 0;
    proof
        .visit_replay_leaves(None, &mut |leaf| {
            assert_eq!(leaf.classes(), [2, 0]);
            assert_eq!(leaf.literal_count(), 0);
            assert_eq!(leaf.literals().count(), 0);
            assert!(!leaf.is_empty());
            assert!(leaf.min_margin().is_finite() && leaf.min_margin() > 0.0);
            visited += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(visited, 1);
    assert_eq!(live(&memory), before);
    drop(session);
    assert_eq!(live(&memory), before);
    drop(proof);
    assert_eq!(live(&memory), 0);
}

#[test]
fn cli_tree_replay_actual_children_export_absolute_head_sites_and_complete_large_class_order() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(66, 17, false, false, 0.75), &budget, options());
    let cert = certificate(original_open(&session, 17));
    // The original target row is [1,-1] and every competitor row is zero.
    // The actual ranker selects the first negative margin coefficient: head1.
    // Check the retained cover independently before inspecting its export.
    for (_, path) in cert.leaves() {
        assert_eq!(
            path.path(None).unwrap().literals()[0].site,
            crate::certify::CoverSite::Head { index: 1 }
        );
    }
    let proof = PreparedRootProof::from_prepared_tree(cert, None).unwrap();
    let stats = proof.replay_stats(None).unwrap();
    assert_eq!(
        (
            stats.site_count,
            stats.leaf_count,
            stats.literal_count,
            stats.max_depth
        ),
        (2, 2, 2, 1)
    );
    let order: Vec<_> = (0..66).filter(|&c| c != 17).rev().collect();
    let before = live(&memory);
    let mut sides = Vec::new();
    proof
        .visit_replay_leaves(None, &mut |leaf| {
            assert_eq!(leaf.classes(), order);
            assert!(!leaf.is_empty());
            assert_eq!(leaf.literal_count(), 1);
            let literals = leaf.literals().collect::<Result<Vec<_>>>()?;
            assert_eq!(literals.len(), 1);
            assert_eq!((literals[0].op(), literals[0].index()), (4, 1));
            sides.push(literals[0].positive());
            assert!(
                leaf.min_margin().is_finite()
                    && leaf.min_margin() > 0.0
                    && leaf.min_margin() <= 0.75
            );
            Ok(())
        })
        .unwrap();
    sides.sort_unstable();
    assert_eq!(sides, [false, true]);
    for quarters in 0..=12 {
        let x = BigRational::new(quarters.into(), 4.into());
        assert_eq!(
            exact_original_margin(x, false, BigRational::new(3.into(), 4.into())),
            BigRational::new(3.into(), 4.into())
        );
    }
    assert_eq!(live(&memory), before);
    drop(proof);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn cli_tree_replay_cancel_during_callback_refuses_completion_and_keeps_original_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&session, 0)), None)
            .unwrap();
    let before = live(&memory);
    let mut visited = 0;
    assert!(proof
        .visit_replay_leaves(None, &mut |leaf| {
            let mut literals = leaf.literals();
            cancel.cancel();
            assert!(literals.next().unwrap().is_err());
            visited += 1;
            Ok(())
        })
        .is_err());
    assert_eq!(visited, 1);
    assert_eq!(live(&memory), before);
    drop(session);
    assert_eq!(live(&memory), before);
    drop(proof);
    assert_eq!(live(&memory), 0);
}

#[test]
fn cli_tree_replay_expired_requested_clock_refuses_before_callbacks_or_new_admission() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&session, 0)), None)
            .unwrap();
    let before = budget.receipt().unwrap().admitted;
    let at = Some(Instant::now());
    let mut visited = 0;
    assert!(proof.replay_stats(at).is_err());
    assert!(proof
        .visit_replay_sites(at, &mut |_, _| {
            visited += 1;
            Ok(())
        })
        .is_err());
    assert!(proof
        .visit_replay_leaves(at, &mut |_| {
            visited += 1;
            Ok(())
        })
        .is_err());
    assert_eq!(visited, 0);
    assert_eq!(budget.receipt().unwrap().admitted, before);
    proof.checkpoint(None).unwrap();
}

#[test]
fn cli_tree_replay_callback_failure_preserves_held_proof_and_same_ledger_retry() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(2, 0, false, false, 0.75), &budget, options());
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&session, 0)), None)
            .unwrap();
    let before = live(&memory);
    assert!(proof
        .visit_replay_leaves(None, &mut |_| Err(ny_core::NyError::InvalidSpec(
            "actual exporter refused".into()
        )))
        .is_err());
    assert_eq!(live(&memory), before);
    assert!(proof.budget().shares_ledger(&budget));
    let mut visited = 0;
    proof
        .visit_replay_leaves(None, &mut |_| {
            visited += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(visited, 2);
    proof.checkpoint(None).unwrap();
    drop(session);
    drop(proof);
    assert_eq!(live(&memory), 0);
}

#[test]
fn cli_tree_debug_keeps_reference_arithmetic_separate_and_preserves_real_owner_claims() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(
        fixture(2, 0, false, false, 0.75),
        &budget,
        PreparedRootProofOptions {
            reference_arithmetic: super::super::PreparedReferenceArithmetic::Binary32Certified,
            ..options()
        },
    );
    let proof =
        PreparedRootProof::from_prepared_tree(certificate(original_open(&session, 0)), None)
            .unwrap();
    let adapter = proof.publication.ordinary().unwrap().adapter();
    let projection_bytes = adapter.resident_bytes();
    let terminal_bytes = proof.publication.terminal().resident_bytes();
    let publication_bytes = proof.publication.resident_bytes();
    assert!(!session.policy().root_f32 && !session.policy().tableau_f32);
    assert!(!adapter.receipt().backward_interm.enabled);
    assert_eq!(adapter.receipt().reference_conv_ops, 1);
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let observed = format!("{proof:?}");
    assert!(observed.contains("source_region: DeclaredBox"));
    assert!(observed.contains("parameter_contract: Binary64WithFoldUncertainty"));
    assert!(observed.contains("Binary32Certified"));
    assert!(observed.contains("root_f32: false"));
    assert!(observed.contains("tableau_f32: false"));
    assert!(observed.contains(&format!(
        "projection_reserved_bytes: Some({projection_bytes})"
    )));
    assert!(observed.contains(&format!("terminal_reserved_bytes: {terminal_bytes}")));
    assert!(observed.contains(&format!("publication_reserved_bytes: {publication_bytes}")));
    assert!(observed.contains("projection_reference_conv_ops: Some(1)"));
    let after = budget.receipt().unwrap();
    assert_eq!(after.work_reserved, before.work_reserved);
    assert_eq!(after.proposals_reserved, before.proposals_reserved);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(live(&memory), bytes);
    proof.checkpoint(None).unwrap();
    drop(session);
    drop(proof);
    assert_eq!(live(&memory), 0);
}
