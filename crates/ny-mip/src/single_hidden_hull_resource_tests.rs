// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_rational::BigRational;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Default)]
struct State {
    attempted: u64,
    work: u64,
    proposals: u64,
    scratch: usize,
    persistent: usize,
}
struct TestLedger {
    state: Mutex<State>,
    cap: usize,
    max_work: u64,
    max_proposals: u64,
    polls: AtomicUsize,
    stop: AtomicUsize,
}
#[derive(Clone)]
struct Authority(Arc<TestLedger>);
struct TestLease {
    ledger: Arc<TestLedger>,
    request: HullWorkRequest,
    live: bool,
}
struct TestClaim {
    ledger: Arc<TestLedger>,
    bytes: usize,
}
impl HullRetainedResource for TestClaim {}
impl Drop for TestClaim {
    fn drop(&mut self) {
        self.ledger.state.lock().unwrap().persistent -= self.bytes;
    }
}
impl TestLedger {
    fn checkpoint(&self) -> Result<(), HullResourceRefusal> {
        let count = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
        if count >= self.stop.load(Ordering::SeqCst) {
            Err(HullResourceRefusal::Cancelled)
        } else {
            Ok(())
        }
    }
}
impl HullQueryAuthority for Authority {
    fn checkpoint(&self, _: HullLeaseClass) -> Result<(), HullResourceRefusal> {
        self.0.checkpoint()
    }
    fn lease(
        &self,
        _: HullLeaseClass,
        request: HullWorkRequest,
    ) -> Result<Box<dyn HullResourceLease>, HullResourceRefusal> {
        self.0.checkpoint()?;
        let mut s = self.0.state.lock().unwrap();
        s.attempted += 1;
        if s.scratch + s.persistent + request.scratch_bytes + request.persistent_bytes > self.0.cap
        {
            return Err(HullResourceRefusal::Memory);
        }
        if s.work + request.work > self.0.max_work
            || s.proposals + request.proposals > self.0.max_proposals
        {
            return Err(HullResourceRefusal::Work);
        }
        s.work += request.work;
        s.proposals += request.proposals;
        s.scratch += request.scratch_bytes;
        s.persistent += request.persistent_bytes;
        Ok(Box::new(TestLease {
            ledger: self.0.clone(),
            request,
            live: true,
        }))
    }
}
impl HullResourceLease for TestLease {
    fn checkpoint(&self) -> Result<(), HullResourceRefusal> {
        self.ledger.checkpoint()
    }
    fn retain(mut self: Box<Self>) -> Result<Arc<dyn HullRetainedResource>, HullResourceRefusal> {
        self.checkpoint()?;
        self.ledger.state.lock().unwrap().scratch -= self.request.scratch_bytes;
        self.live = false;
        Ok(Arc::new(TestClaim {
            ledger: self.ledger.clone(),
            bytes: self.request.persistent_bytes,
        }))
    }
}
impl Drop for TestLease {
    fn drop(&mut self) {
        if self.live {
            let mut s = self.ledger.state.lock().unwrap();
            s.scratch -= self.request.scratch_bytes;
            s.persistent -= self.request.persistent_bytes;
        }
    }
}
fn context(cap: usize, work: u64, proposals: u64) -> (HullQueryContext, Arc<TestLedger>) {
    let ledger = Arc::new(TestLedger {
        state: Mutex::new(State::default()),
        cap,
        max_work: work,
        max_proposals: proposals,
        polls: AtomicUsize::new(0),
        stop: AtomicUsize::new(usize::MAX),
    });
    (HullQueryContext::new(Authority(ledger.clone())), ledger)
}
const W1: [f32; 4] = [1.0, -0.5, -0.25, 1.5];
const W2: [f32; 4] = [-1.0, 0.75, 0.0, 0.0];
const B1: [f64; 2] = [0.125, -0.375];
const OBJECTIVE: [(usize, f64); 2] = [(0, 1.0), (1, -1.0)];
const LOWER: [f32; 2] = [-1.0, -0.75];
const UPPER: [f32; 2] = [0.5, 1.0];
fn data(b2: &[f64]) -> HullOriginalData<'_> {
    HullOriginalData {
        input_dim: 2,
        hidden_dim: 2,
        output_dim: 2,
        w1: &W1,
        w2: &W2,
        b1: HullBiasData::Binary64(&B1),
        b2: HullBiasData::Binary64(b2),
        objective: &OBJECTIVE,
        objective_constant: 0.0,
        lower: &LOWER,
        upper: &UPPER,
    }
}
fn rat(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}

fn policy_scores(p: &HullMarginProblem) -> (HullMultipliers, f64, HullMultipliers, f64) {
    let bounds = preactivation_bounds(p);
    let roles = classify(p, &bounds);
    let legacy = triangle_multipliers(p, &roles, &bounds);
    let area = triangle_multipliers_with_policy_polled(
        p,
        &roles,
        &bounds,
        PositiveLowerSlopePolicy::Area,
        &mut || true,
    )
    .unwrap();
    let legacy_score = certify_hull_lower_bound(p, &legacy).unwrap();
    let area_score = certify_hull_lower_bound(p, &area).unwrap();
    (legacy, legacy_score, area, area_score)
}

fn assert_selected_and_replayed(
    owner: &HullOriginalOwner,
    context: &HullQueryContext,
    checked: &HullTriangleResult,
) -> (f64, f64) {
    let (legacy, legacy_score, area, area_score) = policy_scores(&owner.0.problem);
    let (expected_policy, expected, score) = if area_score > legacy_score {
        (PositiveLowerSlopePolicy::Area, area, area_score)
    } else {
        (PositiveLowerSlopePolicy::Proportional, legacy, legacy_score)
    };
    assert_eq!(*checked.certificate.0.multipliers, expected);
    assert_eq!(checked.certificate.lower_bound().to_bits(), score.to_bits());
    assert!(checked.certificate.lower_bound() >= legacy_score);
    assert_eq!(
        checked.certificate.triangle_search(),
        HullTriangleSearchStats {
            search_attempts: 2,
            search_evaluations: 2,
            certificate_proposals: 1,
            certificate_evaluations: 1,
            selected_policy: expected_policy,
        }
    );
    let replay = replay_hull_original(&checked.certificate, context).unwrap();
    assert_eq!(replay.origin(), HullProofOrigin::OriginalReplay);
    assert_eq!(replay.scope(), owner.scope());
    assert_eq!(replay.lower_bound().to_bits(), score.to_bits());
    assert_eq!(
        replay.triangle_search(),
        checked.certificate.triangle_search()
    );
    (legacy_score, area_score)
}

const ONE_OBJECTIVE: [(usize, f64); 1] = [(0, 1.0)];
fn one_dimensional_data<'a>(
    w1: &'a [f32],
    b1: &'a [f64],
    w2: &'a [f32],
    b2: &'a [f64],
    lower: &'a [f32; 1],
    upper: &'a [f32; 1],
) -> HullOriginalData<'a> {
    HullOriginalData {
        input_dim: 1,
        hidden_dim: b1.len(),
        output_dim: 1,
        w1,
        w2,
        b1: HullBiasData::Binary64(b1),
        b2: HullBiasData::Binary64(b2),
        objective: &ONE_OBJECTIVE,
        objective_constant: 0.0,
        lower,
        upper,
    }
}

// Exact endpoints and every exact ReLU kink exhaust the minima of a 1D
// piecewise-affine original network. This oracle does not use slope proposals.
fn exact_original_minimum_1d(owner: &HullOriginalOwner) -> BigRational {
    let p = &owner.0.problem;
    assert_eq!((p.input_dim, owner.0.output_dim), (1, 1));
    assert_eq!(owner.0.objective, ONE_OBJECTIVE);
    let lo = rat(p.lower[0]);
    let hi = rat(p.upper[0]);
    let mut points = vec![lo.clone(), hi.clone()];
    for j in 0..p.hidden_dim {
        let w = rat(p.w1[j]);
        if w != rat(0.0) {
            let kink = -rat(p.b1[j]) / w;
            if kink >= lo && kink <= hi {
                points.push(kink);
            }
        }
    }
    points.sort();
    points.dedup();
    points
        .into_iter()
        .map(|x| {
            let mut value = rat(owner.0.b2[0]) + rat(owner.0.original_constant);
            for j in 0..p.hidden_dim {
                let z = rat(p.w1[j]) * &x + rat(p.b1[j]);
                value += rat(owner.0.w2[j]) * z.max(rat(0.0));
            }
            value
        })
        .min()
        .unwrap()
}

#[test]
fn hull_resource_original_triangle_and_replay_match_existing_checker_and_rational_vertices() {
    let (context, ledger) = context(1 << 20, 1 << 30, 4);
    let owner = import_hull_original(data(&[4.0, 0.0]), &context).unwrap();
    let checked = checked_hull_triangle(&owner, &context).unwrap();
    assert!(checked.require_strict_positive().is_ok());
    let p = &owner.0.problem;
    let (legacy, legacy_score, area, area_score) = policy_scores(p);
    let m = if area_score > legacy_score {
        area
    } else {
        legacy
    };
    assert_eq!(*checked.certificate.0.multipliers, m);
    assert_eq!(
        checked.certificate.lower_bound().to_bits(),
        certify_hull_lower_bound(p, &m).unwrap().to_bits()
    );
    for x0 in [-1.0_f64, -0.25, 0.5] {
        for x1 in [-0.75_f64, 0.125, 1.0] {
            let x = [rat(x0), rat(x1)];
            let mut original = rat(4.0);
            for j in 0..2 {
                let mut z = rat(B1[j]);
                for i in 0..2 {
                    z += rat(ny_core::f32_to_f64_exact(W1[j * 2 + i])) * &x[i];
                }
                original += rat(ny_core::f32_to_f64_exact(W2[j])) * z.max(rat(0.0));
            }
            assert!(rat(checked.certificate.lower_bound()) <= original);
        }
    }
    let replay = replay_hull_original(&checked.certificate, &context).unwrap();
    assert_eq!(replay.origin(), HullProofOrigin::OriginalReplay);
    assert_eq!(replay.scope(), owner.scope());
    assert_eq!(
        replay.lower_bound().to_bits(),
        checked.certificate.lower_bound().to_bits()
    );
    let s = ledger.state.lock().unwrap();
    assert_eq!(s.attempted, 3);
    assert_eq!(s.proposals, 1);
    assert_eq!(s.scratch, 0);
    assert!(s.work > 0 && s.persistent > 0);
}

#[test]
fn hull_resource_policy_legacy_exact_bits_and_area_signed_tie_rule() {
    let p = HullMarginProblem::new(
        vec![1.0, -1.0, -2.0, 0.25, 0.25],
        vec![0.5, -0.5, 0.25, 1.0, -1.0],
        1,
        &[1.0, 2.0, -0.5, -1.0, 3.0],
        &[4.0],
        1,
        &ONE_OBJECTIVE,
        0.0,
        vec![-1.0],
        vec![1.0],
    )
    .unwrap();
    let (legacy, _, area, _) = policy_scores(&p);
    assert_eq!(
        legacy.alpha.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [0.75_f64, 0.25, 0.0, 1.0, 0.0].map(f64::to_bits)
    );
    assert_eq!(
        legacy.g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [0.0_f64, 0.0, -1.125, 0.25, 0.0].map(f64::to_bits)
    );
    assert_eq!(area.alpha, [1.0, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(area.g, legacy.g);
    // Equal positive and negative areas choose alpha=1; unequal signed
    // bounds above choose opposite endpoints. All legacy callers stay .5.
    let bounds = [(-1.0, 1.0)];
    let roles = [Role::PosUnstable];
    let tie = triangle_multipliers_with_policy_polled(
        &p,
        &roles,
        &bounds,
        PositiveLowerSlopePolicy::Area,
        &mut || true,
    )
    .unwrap();
    assert_eq!(tie.alpha[0], 1.0);
    assert_eq!(triangle_multipliers(&p, &roles, &bounds).alpha[0], 0.5);
}

#[test]
fn hull_resource_paid_search_area_zero_and_one_gain_reach_original_rational_replay() {
    // First: 5/4+relu(x)-(1/4)relu(2-x), x in [-2,1].
    // Second: 9/4+relu(x)-relu(x+2), x in [-1,2]. Both exact minima are 1/4.
    for (w1, b1, w2, b2, lower, upper, expected_alpha) in [
        (
            [1.0_f32, -1.0],
            [0.0_f64, 2.0],
            [1.0_f32, -0.25],
            [1.25_f64],
            [-2.0_f32],
            [1.0_f32],
            0.0_f64,
        ),
        (
            [1.0_f32, 1.0],
            [0.0_f64, 2.0],
            [1.0_f32, -1.0],
            [2.25_f64],
            [-1.0_f32],
            [2.0_f32],
            1.0_f64,
        ),
    ] {
        let (context, ledger) = context(1 << 20, 1 << 30, 1);
        let owner = import_hull_original(
            one_dimensional_data(&w1, &b1, &w2, &b2, &lower, &upper),
            &context,
        )
        .unwrap();
        let checked = checked_hull_triangle(&owner, &context).unwrap();
        let (legacy, area) = assert_selected_and_replayed(&owner, &context, &checked);
        assert!(legacy < 0.0 && area > 0.0);
        assert!(checked.require_strict_positive().is_ok());
        assert_eq!(checked.certificate.0.multipliers.alpha[0], expected_alpha);
        let exact = exact_original_minimum_1d(&owner);
        assert_eq!(exact, rat(0.25));
        assert!(rat(checked.certificate.lower_bound()) <= exact);
        let plan =
            hull_ingress_plan(one_dimensional_data(&w1, &b1, &w2, &b2, &lower, &upper)).unwrap();
        let s = ledger.state.lock().unwrap();
        assert_eq!(s.proposals, 1);
        assert_eq!(
            s.work,
            plan.originals.work + plan.triangle.work + plan.replay.work
        );
        assert_eq!(s.scratch, 0);
    }
}

#[test]
fn hull_resource_paid_search_preserves_useful_legacy_cancellation_proof() {
    // The actual function is constant3/4. Area loses the proportional
    // cancellation and returns a negative score; selecting Area alone regresses.
    let (context, ledger) = context(1 << 20, 1 << 30, 1);
    let source = one_dimensional_data(
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[1.0, -1.0],
        &[0.75],
        &[-1.0],
        &[1.0],
    );
    let owner = import_hull_original(source, &context).unwrap();
    let checked = checked_hull_triangle(&owner, &context).unwrap();
    let (legacy, area) = assert_selected_and_replayed(&owner, &context, &checked);
    assert!(legacy > 0.0 && area < 0.0);
    assert!(checked.require_strict_positive().is_ok());
    assert_eq!(
        checked.certificate.triangle_search().selected_policy,
        PositiveLowerSlopePolicy::Proportional
    );
    assert_eq!(exact_original_minimum_1d(&owner), rat(0.75));
    assert!(rat(checked.certificate.lower_bound()) <= rat(0.75));
    assert_eq!(ledger.state.lock().unwrap().proposals, 1);
}

#[test]
fn hull_resource_paid_search_signed_zero_score_tie_and_near_zero_keep_original_authority() {
    for constant in [
        f64::from_bits(0.5_f64.to_bits() - 1),
        0.5,
        f64::from_bits(0.5_f64.to_bits() + 1),
    ] {
        let (context, _) = context(1 << 20, 1 << 30, 1);
        let b2 = [constant];
        let owner = import_hull_original(
            one_dimensional_data(&[1.0, 1.0], &[0.0, 0.0], &[1.0, -1.0], &b2, &[-1.0], &[1.0]),
            &context,
        )
        .unwrap();
        let checked = checked_hull_triangle(&owner, &context).unwrap();
        let (legacy, _) = assert_selected_and_replayed(&owner, &context, &checked);
        assert_eq!(
            checked.certificate.lower_bound().to_bits(),
            legacy.to_bits()
        );
        if constant <= 0.5 {
            assert!(!checked.certificate.verified());
        }
        assert!(rat(checked.certificate.lower_bound()) <= exact_original_minimum_1d(&owner));
    }
    // Dead signed-zero neuron: both proposal scores have identical bits.
    // The true margin is exactly zero; neither preview can turn a tie into proof.
    let (context, _) = context(1 << 20, 1 << 30, 1);
    let owner = import_hull_original(
        one_dimensional_data(&[-0.0], &[-0.0], &[1.0], &[-0.0], &[-0.0], &[0.0]),
        &context,
    )
    .unwrap();
    let checked = checked_hull_triangle(&owner, &context).unwrap();
    let (legacy, area) = assert_selected_and_replayed(&owner, &context, &checked);
    assert_eq!(legacy.to_bits(), area.to_bits());
    assert_eq!(
        checked.certificate.triangle_search().selected_policy,
        PositiveLowerSlopePolicy::Proportional
    );
    assert!(!checked.certificate.verified());
    assert_eq!(exact_original_minimum_1d(&owner), rat(0.0));
}

#[test]
fn hull_resource_paid_search_invalid_area_preserves_completed_positive_legacy_score() {
    // The original function is the positive constant1.1e308. Proportional
    // multipliers cancel the two signed ReLU terms without overflow; Area's
    // alpha1 makes its positive coefficient overflow. That optional invalid
    // advice must not discard the completed usable legacy proof.
    let objective = [(0, 1.0e308)];
    let mut source = one_dimensional_data(
        &[2.0, 2.0],
        &[0.0, 0.0],
        &[1.0, -1.0],
        &[0.0],
        &[-1.0],
        &[1.0],
    );
    source.objective = &objective;
    source.objective_constant = 1.1e308;
    let (context, ledger) = context(1 << 20, 1 << 30, 1);
    let owner = import_hull_original(source, &context).unwrap();
    let p = &owner.0.problem;
    let bounds = preactivation_bounds(p);
    let roles = classify(p, &bounds);
    let legacy = triangle_multipliers(p, &roles, &bounds);
    let legacy_score = certify_hull_lower_bound(p, &legacy).unwrap();
    let area = triangle_multipliers_with_policy_polled(
        p,
        &roles,
        &bounds,
        PositiveLowerSlopePolicy::Area,
        &mut || true,
    )
    .unwrap();
    assert!(certify_hull_lower_bound(p, &area).is_none());
    assert!(legacy_score > 0.0);
    let checked = checked_hull_triangle(&owner, &context).unwrap();
    assert_eq!(*checked.certificate.0.multipliers, legacy);
    assert_eq!(
        checked.certificate.lower_bound().to_bits(),
        legacy_score.to_bits()
    );
    assert_eq!(
        checked.certificate.triangle_search(),
        HullTriangleSearchStats {
            search_attempts: 2,
            search_evaluations: 1,
            certificate_proposals: 1,
            certificate_evaluations: 1,
            selected_policy: PositiveLowerSlopePolicy::Proportional,
        }
    );
    let replay = replay_hull_original(&checked.certificate, &context).unwrap();
    assert!(replay.verified());
    assert_eq!(replay.lower_bound().to_bits(), legacy_score.to_bits());
    assert!(rat(replay.lower_bound()) <= rat(1.1e308));
    assert_eq!(ledger.state.lock().unwrap().proposals, 1);
}

#[test]
fn hull_resource_paid_search_work_refusal_happens_before_evaluation_and_one_grant_stays_spent() {
    let source = data(&[4.0, 0.0]);
    let plan = hull_ingress_plan(source).unwrap();
    let old_triangle_work = 64 * 4 + 128 * 2 + 64 * 2 + 4096;
    assert!(plan.triangle.work > old_triangle_work);
    let (context, ledger) = context(1 << 20, plan.originals.work + old_triangle_work, 1);
    let owner = import_hull_original(source, &context).unwrap();
    let original_live = ledger.state.lock().unwrap().persistent;
    assert!(matches!(
        checked_hull_triangle(&owner, &context),
        Err(HullQueryRefusal::Resource(HullResourceRefusal::Work))
    ));
    let s = ledger.state.lock().unwrap();
    assert_eq!(
        (s.work, s.proposals, s.scratch, s.persistent),
        (plan.originals.work, 0, 0, original_live)
    );
    drop(s);

    let (context, ledger) = self::context(1 << 20, 1 << 30, 1);
    let owner = import_hull_original(source, &context).unwrap();
    let checked = checked_hull_triangle(&owner, &context).unwrap();
    let spent = ledger.state.lock().unwrap().work;
    assert!(matches!(
        checked_hull_triangle(&owner, &context),
        Err(HullQueryRefusal::Resource(HullResourceRefusal::Work))
    ));
    assert_eq!(ledger.state.lock().unwrap().work, spent);
    assert_eq!(ledger.state.lock().unwrap().proposals, 1);
    assert!(checked.certificate.verified());
}

#[test]
fn hull_resource_paid_search_cancellation_releases_only_new_claims_and_keeps_spent_grant() {
    // Determine the actual query poll span on the same fixed arithmetic, then
    // stop another admitted query during its search. No candidate certificate
    // is published, the immutable original remains and no allowance resets.
    let (completed_context, completed_ledger) = context(1 << 20, 1 << 30, 1);
    let owner = import_hull_original(data(&[4.0, 0.0]), &completed_context).unwrap();
    let start = completed_ledger.polls.load(Ordering::SeqCst);
    let completed = checked_hull_triangle(&owner, &completed_context).unwrap();
    let span = completed_ledger.polls.load(Ordering::SeqCst) - start;
    assert_eq!(
        completed.certificate.triangle_search().search_evaluations,
        2
    );
    assert!(span > 30);
    let (context, ledger) = context(1 << 20, 1 << 30, 1);
    let owner = import_hull_original(data(&[4.0, 0.0]), &context).unwrap();
    let original_live = ledger.state.lock().unwrap().persistent;
    let prior_work = ledger.state.lock().unwrap().work;
    let stop = ledger.polls.load(Ordering::SeqCst) + span / 2;
    ledger.stop.store(stop, Ordering::SeqCst);
    assert!(matches!(
        checked_hull_triangle(&owner, &context),
        Err(HullQueryRefusal::Resource(HullResourceRefusal::Cancelled))
    ));
    let state = ledger.state.lock().unwrap();
    assert_eq!(
        (state.scratch, state.persistent, state.proposals),
        (0, original_live, 1)
    );
    assert!(state.work > prior_work);
    assert_eq!(owner.scope(), completed.certificate.scope());
}

#[test]
fn hull_resource_last_clone_holds_original_query_and_replay_claims_without_work_refill() {
    let (context, ledger) = context(1 << 20, 1 << 30, 1);
    let owner = import_hull_original(data(&[4.0, 0.0]), &context).unwrap();
    let checked = checked_hull_triangle(&owner, &context).unwrap();
    let replay = replay_hull_original(&checked.certificate, &context).unwrap();
    let clone = replay.clone();
    let before = ledger.state.lock().unwrap().persistent;
    drop(owner);
    drop(checked);
    drop(replay);
    assert_eq!(ledger.state.lock().unwrap().persistent, before);
    let spent = ledger.state.lock().unwrap().work;
    assert!(matches!(
        checked_hull_triangle(&clone.original(), &context),
        Err(HullQueryRefusal::Resource(HullResourceRefusal::Work))
    ));
    assert_eq!(ledger.state.lock().unwrap().work, spent);
    drop(clone);
    let s = ledger.state.lock().unwrap();
    assert_eq!((s.scratch, s.persistent), (0, 0));
    assert_eq!(s.work, spent);
}

#[test]
fn hull_resource_foreign_context_cannot_refresh_query_or_replay_allowance() {
    let (context, ledger) = context(1 << 20, 1 << 30, 4);
    let owner = import_hull_original(data(&[4.0, 0.0]), &context).unwrap();
    let certificate = checked_hull_triangle(&owner, &context).unwrap().certificate;
    let (foreign, other) = self::context(1 << 20, 1 << 30, 4);
    assert!(matches!(
        checked_hull_triangle(&owner, &foreign),
        Err(HullQueryRefusal::ForeignAuthority)
    ));
    assert!(matches!(
        replay_hull_original(&certificate, &foreign),
        Err(HullQueryRefusal::ForeignAuthority)
    ));
    assert_eq!(other.state.lock().unwrap().attempted, 0);
    assert_eq!(ledger.state.lock().unwrap().attempted, 2);
}

#[test]
fn hull_resource_shape_and_capacity_refuse_before_original_scans_and_copies() {
    let (context, ledger) = context(1, 1 << 30, 4);
    let mut invalid = data(&[4.0, 0.0]);
    invalid.w1 = &[f32::NAN];
    assert!(matches!(
        import_hull_original(invalid, &context),
        Err(HullQueryRefusal::Original(HullDecline::ShapeMismatch))
    ));
    assert_eq!(ledger.state.lock().unwrap().attempted, 0);
    assert!(matches!(
        import_hull_original(data(&[4.0, 0.0]), &context),
        Err(HullQueryRefusal::Resource(HullResourceRefusal::Memory))
    ));
    let s = ledger.state.lock().unwrap();
    assert_eq!(s.attempted, 1);
    assert_eq!((s.work, s.scratch, s.persistent), (0, 0, 0));
}

#[test]
fn hull_resource_cancel_during_real_import_releases_capacity_but_consumes_work() {
    let (context, ledger) = context(1 << 20, 1 << 30, 4);
    // Constructor/context/lease polls admit first; stop during actual W1/W2 import.
    ledger.stop.store(12, Ordering::SeqCst);
    assert!(matches!(
        import_hull_original(data(&[4.0, 0.0]), &context),
        Err(HullQueryRefusal::Resource(HullResourceRefusal::Cancelled))
    ));
    let s = ledger.state.lock().unwrap();
    assert_eq!(s.attempted, 1);
    assert!(s.work > 0);
    assert_eq!((s.scratch, s.persistent), (0, 0));
}

#[test]
fn hull_resource_tie_sat_and_mutable_report_never_mint_a_positive_sealed_claim() {
    let w1 = [1.0_f32];
    let w2 = [-1.0_f32];
    let lower = [-1.0_f32];
    let upper = [1.0_f32];
    let objective = [(0, 1.0)];
    for constant in [1.0_f64, 0.5] {
        let (context, _) = context(1 << 20, 1 << 30, 4);
        let b2 = [constant];
        let source = HullOriginalData {
            input_dim: 1,
            hidden_dim: 1,
            output_dim: 1,
            w1: &w1,
            w2: &w2,
            b1: HullBiasData::Zeros,
            b2: HullBiasData::Binary64(&b2),
            objective: &objective,
            objective_constant: 0.0,
            lower: &lower,
            upper: &upper,
        };
        let owner = import_hull_original(source, &context).unwrap();
        let mut checked = checked_hull_triangle(&owner, &context).unwrap();
        assert!(!checked.certificate.verified());
        assert!(matches!(
            checked.require_strict_positive(),
            Err(HullQueryRefusal::UnpricedLpOrBab)
        ));
        checked.report.certified = Some(999.0);
        checked.report.lp_status = HullLpStatus::Solved;
        assert!(checked.report.verified());
        assert!(!checked.certificate.verified());
        let replay = replay_hull_original(&checked.certificate, &context).unwrap();
        assert!(!replay.verified());
        assert!(replay.lower_bound() <= constant - 1.0);
    }
}

#[test]
fn hull_resource_original_poison_scope_objective_and_sealed_bound_refuse_replay() {
    for field in 0..5 {
        let (context, _) = context(1 << 20, 1 << 30, 4);
        let owner = import_hull_original(data(&[4.0, 0.0]), &context).unwrap();
        let mut certificate = checked_hull_triangle(&owner, &context).unwrap().certificate;
        if field == 4 {
            Arc::get_mut(&mut certificate.0).unwrap().bound += 1.0;
        } else {
            drop(owner);
            let proof = Arc::get_mut(&mut certificate.0).unwrap();
            let original = Arc::get_mut(&mut proof.original.0).unwrap();
            match field {
                0 => original.w2[0] = -2.0,
                1 => original.problem.lower[0] = -0.5,
                2 => original.objective[0] = (1, 1.0),
                3 => original.problem.c[0] = 3.0,
                _ => unreachable!(),
            }
        }
        assert!(matches!(
            replay_hull_original(&certificate, &context),
            Err(HullQueryRefusal::ReplayMismatch)
        ));
    }
}

#[test]
fn hull_resource_cancel_during_final_replay_publication_keeps_prior_proof_and_spent_work() {
    let (context, ledger) = context(1 << 20, 1 << 30, 4);
    let owner = import_hull_original(data(&[4.0, 0.0]), &context).unwrap();
    let certificate = checked_hull_triangle(&owner, &context).unwrap().certificate;
    let start = ledger.polls.load(Ordering::SeqCst);
    let first = replay_hull_original(&certificate, &context).unwrap();
    let finish = ledger.polls.load(Ordering::SeqCst);
    let replay_polls = finish - start;
    assert!(replay_polls > 10);
    drop(first);
    let persistent = ledger.state.lock().unwrap().persistent;
    let spent = ledger.state.lock().unwrap().work;
    ledger.stop.store(finish + replay_polls, Ordering::SeqCst);
    // Identical original replay reaches its real final checkpoint only after
    // scope/head reduction/full closed-form replay and claim transfer.
    assert!(matches!(
        replay_hull_original(&certificate, &context),
        Err(HullQueryRefusal::Resource(HullResourceRefusal::Cancelled))
    ));
    let state = ledger.state.lock().unwrap();
    assert_eq!(state.attempted, 4);
    assert_eq!((state.scratch, state.persistent), (0, persistent));
    assert!(state.work > spent);
    assert_eq!(certificate.origin(), HullProofOrigin::CheckedQuery);
    assert!(certificate.verified());
}
