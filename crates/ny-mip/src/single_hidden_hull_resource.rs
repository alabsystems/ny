// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Caller-resource ingress for the existing original single-hidden checker.
//!
//! This bounded profile searches two triangle slope policies and submits one
//! finalized multiplier proposal. Its two directed score previews grant no
//! certificate or verdict; all three evaluations are prepaid. The original
//! directed closed form remains the arithmetic authority. AY LP sessions and
//! BaB do not have a qualified workspace/cancellation contract for this profile;
//! an undecided triangle requests that distinct, unqualified route explicitly.
//! Raw original backing storage is caller-owned. Its logical import is charged,
//! and every owned copy/evidence capacity stays leased through its last clone.

use super::*;
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HullLeaseClass {
    Optional,
    Replay,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HullWorkRequest {
    pub scratch_bytes: usize,
    pub persistent_bytes: usize,
    pub work: u64,
    pub proposals: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HullResourceRefusal {
    Deadline,
    Cancelled,
    Memory,
    Work,
    Overflow,
    Unavailable,
}

/// Resource operations grant no arithmetic/proof authority. Implementations
/// adapt an existing caller ledger; they must not create a new allowance.
pub trait HullQueryAuthority: Send + Sync {
    fn lease(
        &self,
        class: HullLeaseClass,
        request: HullWorkRequest,
    ) -> Result<Box<dyn HullResourceLease>, HullResourceRefusal>;
    fn checkpoint(&self, class: HullLeaseClass) -> Result<(), HullResourceRefusal>;
}

pub trait HullResourceLease: Send + Sync {
    fn checkpoint(&self) -> Result<(), HullResourceRefusal>;
    /// Complete before publication and release scratch, retaining exactly the
    /// admitted persistent capacity until the returned last-owner claim drops.
    fn retain(self: Box<Self>) -> Result<Arc<dyn HullRetainedResource>, HullResourceRefusal>;
}

pub trait HullRetainedResource: Send + Sync {}

/// Clone shares this exact authority. Equal limits or a different adapter do
/// not authorize replay/refresh on a new ledger.
#[derive(Clone)]
pub struct HullQueryContext(Arc<dyn HullQueryAuthority>);
impl HullQueryContext {
    pub fn new(authority: impl HullQueryAuthority + 'static) -> Self {
        Self(Arc::new(authority))
    }
    pub fn shares_authority(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub fn checkpoint(&self, class: HullLeaseClass) -> Result<(), HullQueryRefusal> {
        self.0
            .checkpoint(class)
            .map_err(HullQueryRefusal::Resource)?;
        if ny_core::instance_budget::deadline().is_some_and(|d| Instant::now() >= d) {
            return Err(HullQueryRefusal::Resource(HullResourceRefusal::Deadline));
        }
        require_hull_environment().map_err(HullQueryRefusal::Original)
    }
    fn lease(
        &self,
        class: HullLeaseClass,
        request: HullWorkRequest,
    ) -> Result<Box<dyn HullResourceLease>, HullQueryRefusal> {
        self.checkpoint(class)?;
        self.0
            .lease(class, request)
            .map_err(HullQueryRefusal::Resource)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HullQueryRefusal {
    Original(HullDecline),
    Resource(HullResourceRefusal),
    ForeignAuthority,
    ReplayMismatch,
    Numerical,
    UnpricedLpOrBab,
}
impl std::fmt::Display for HullQueryRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for HullQueryRefusal {}

/// Original bias sidecar. Binary32 imports widen by bits; a Binary64 sidecar
/// is taken as exact original data, as in the existing folded-MIP contract.
#[derive(Clone, Copy)]
pub enum HullBiasData<'a> {
    Zeros,
    Binary32(&'a [f32]),
    Binary64(&'a [f64]),
}
impl HullBiasData<'_> {
    fn shape(self, len: usize) -> bool {
        match self {
            Self::Zeros => true,
            Self::Binary32(v) => v.len() == len,
            Self::Binary64(v) => v.len() == len,
        }
    }
    fn at(self, i: usize) -> f64 {
        match self {
            Self::Zeros => 0.0,
            Self::Binary32(v) => ny_core::f32_to_f64_exact(v[i]),
            Self::Binary64(v) => v[i],
        }
    }
}

/// Borrowed, exact canonical operands. Shape/spec checks require no copies or
/// coefficient scans. There is no Conv unfold, constant fold or guessed layout
/// in this ingress; callers must separately admit any such preprocessing.
#[derive(Clone, Copy)]
pub struct HullOriginalData<'a> {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub w1: &'a [f32],
    pub b1: HullBiasData<'a>,
    pub w2: &'a [f32],
    pub b2: HullBiasData<'a>,
    pub objective: &'a [(usize, f64)],
    pub objective_constant: f64,
    pub lower: &'a [f32],
    pub upper: &'a [f32],
}

impl HullOriginalData<'_> {
    fn shape(self) -> Result<GeometryPlan, HullQueryRefusal> {
        let (n, h, o) = (self.input_dim, self.hidden_dim, self.output_dim);
        if n == 0 || h == 0 || o == 0 || self.objective.is_empty() {
            return Err(HullQueryRefusal::Original(HullDecline::EmptyDimension));
        }
        if n > HULL_MAX_INPUT_DIM || h > HULL_MAX_HIDDEN_DIM {
            return Err(HullQueryRefusal::Original(HullDecline::SizeCap));
        }
        let first = mul(n, h)?;
        let second = mul(h, o)?;
        if self.w1.len() != first
            || self.w2.len() != second
            || !self.b1.shape(h)
            || !self.b2.shape(o)
            || self.lower.len() != n
            || self.upper.len() != n
        {
            return Err(HullQueryRefusal::Original(HullDecline::ShapeMismatch));
        }
        GeometryPlan::new(n, h, o, self.objective.len())
    }
}

fn overflow() -> HullQueryRefusal {
    HullQueryRefusal::Resource(HullResourceRefusal::Overflow)
}
fn add(a: usize, b: usize) -> Result<usize, HullQueryRefusal> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize, HullQueryRefusal> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn work(n: usize) -> Result<u64, HullQueryRefusal> {
    u64::try_from(n).map_err(|_| overflow())
}

/// Admission capacity estimates, not RSS or an AY allocator quota.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HullIngressPlan {
    pub originals: HullWorkRequest,
    pub triangle: HullWorkRequest,
    pub replay: HullWorkRequest,
}
struct GeometryPlan {
    n: usize,
    h: usize,
    o: usize,
    s: usize,
    requests: HullIngressPlan,
}
impl GeometryPlan {
    fn new(n: usize, h: usize, o: usize, s: usize) -> Result<Self, HullQueryRefusal> {
        let hn = mul(h, n)?;
        let oh = mul(o, h)?;
        let originals = add(add(add(add(hn, oh)?, mul(h, 3)?)?, o)?, mul(n, 2)?)?;
        let persistent = add(add(mul(originals, 8)?, mul(s, 16)?)?, 4096)?;
        let import_work = add(
            add(mul(add(originals, s)?, 16)?, mul(mul(s, h)?, 16)?)?,
            4096,
        )?;
        let multiplier_bytes = add(add(mul(hn, 8)?, mul(h, 8)?)?, 1024)?;
        // Logical scalar work, not a runtime prediction. Each unchanged score
        // pass costs <=32HN+32H+32N+1024: a negative coefficient has one
        // finite scan, <=16 operations for both majorant branches and four
        // coefficient operations; the positive case is cheaper. Array init,
        // validation and final four-corner minimum fit the 32N term. Three
        // passes cost 96HN+96H+96N+3072. Prebounds (6HN), g initialization/
        // live copy/unstable fill (<=3HN), role/count scans and alpha save/
        // replace/restore (<=32H), plus checkpoints and metadata fit the
        // remaining 32HN+160H+32N+9216. Replay has its own unchanged request.
        let triangle_work = add(add(add(mul(hn, 128)?, mul(h, 256)?)?, mul(n, 128)?)?, 12288)?;
        let replay_work = add(
            add(
                add(mul(hn, 32)?, mul(mul(s, h)?, 16)?)?,
                mul(add(originals, s)?, 16)?,
            )?,
            4096,
        )?;
        Ok(Self {
            n,
            h,
            o,
            s,
            requests: HullIngressPlan {
                originals: HullWorkRequest {
                    scratch_bytes: 0,
                    persistent_bytes: persistent,
                    work: work(import_work)?,
                    proposals: 0,
                },
                triangle: HullWorkRequest {
                    scratch_bytes: add(add(mul(h, 32)?, mul(n, 16)?)?, 4096)?,
                    persistent_bytes: multiplier_bytes,
                    work: work(triangle_work)?,
                    proposals: 1,
                },
                replay: HullWorkRequest {
                    scratch_bytes: add(add(mul(h, 8)?, mul(n, 16)?)?, 4096)?,
                    persistent_bytes: 1024,
                    work: work(replay_work)?,
                    proposals: 0,
                },
            },
        })
    }
}

pub fn hull_ingress_plan(data: HullOriginalData<'_>) -> Result<HullIngressPlan, HullQueryRefusal> {
    Ok(data.shape()?.requests)
}

struct OriginalOwner {
    problem: HullMarginProblem,
    w2: Vec<f64>,
    b2: Vec<f64>,
    objective: Vec<(usize, f64)>,
    original_constant: f64,
    output_dim: usize,
    scope: [u8; 32],
    context: HullQueryContext,
    _claim: Arc<dyn HullRetainedResource>,
}
#[derive(Clone)]
pub struct HullOriginalOwner(Arc<OriginalOwner>);
impl HullOriginalOwner {
    pub fn scope(&self) -> [u8; 32] {
        self.0.scope
    }
    pub fn context(&self) -> HullQueryContext {
        self.0.context.clone()
    }
}

fn poll(
    lease: &dyn HullResourceLease,
    context: &HullQueryContext,
    class: HullLeaseClass,
) -> Result<(), HullQueryRefusal> {
    lease.checkpoint().map_err(HullQueryRefusal::Resource)?;
    context.checkpoint(class)
}
fn copy(
    len: usize,
    mut at: impl FnMut(usize) -> f64,
    mut checkpoint: impl FnMut() -> Result<(), HullQueryRefusal>,
) -> Result<Vec<f64>, HullQueryRefusal> {
    checkpoint()?;
    let mut values = Vec::with_capacity(len);
    for i in 0..len {
        if i % 1024 == 0 {
            checkpoint()?;
        }
        let value = at(i);
        if !value.is_finite() {
            return Err(HullQueryRefusal::Original(HullDecline::NonFiniteParameter));
        }
        values.push(value);
    }
    checkpoint()?;
    Ok(values)
}

/// Lease before allocation/import; immutable ownership and original scope are
/// retained together. Existing original construction rounds c/d downward.
pub fn import_hull_original(
    data: HullOriginalData<'_>,
    context: &HullQueryContext,
) -> Result<HullOriginalOwner, HullQueryRefusal> {
    context.checkpoint(HullLeaseClass::Optional)?;
    let plan = data.shape()?;
    let lease = context.lease(HullLeaseClass::Optional, plan.requests.originals)?;
    let check = || poll(&*lease, context, HullLeaseClass::Optional);
    let w1 = copy(
        data.w1.len(),
        |i| ny_core::f32_to_f64_exact(data.w1[i]),
        check,
    )?;
    let w2 = copy(
        data.w2.len(),
        |i| ny_core::f32_to_f64_exact(data.w2[i]),
        check,
    )?;
    let b1 = copy(plan.h, |i| data.b1.at(i), check)?;
    let b2 = copy(plan.o, |i| data.b2.at(i), check)?;
    let lower = copy(plan.n, |i| ny_core::f32_to_f64_exact(data.lower[i]), check)?;
    let upper = copy(plan.n, |i| ny_core::f32_to_f64_exact(data.upper[i]), check)?;
    check()?;
    let mut objective = Vec::with_capacity(plan.s);
    for (i, &value) in data.objective.iter().enumerate() {
        if i % 1024 == 0 {
            check()?;
        }
        objective.push(value);
    }
    let mut stopped = None;
    let mut checkpoint = || match check() {
        Ok(()) => true,
        Err(e) => {
            stopped = Some(e);
            false
        }
    };
    let problem = HullMarginProblem::new_polled(
        w1,
        b1,
        plan.n,
        &w2,
        &b2,
        plan.o,
        &objective,
        data.objective_constant,
        lower,
        upper,
        &mut checkpoint,
    );
    if let Some(e) = stopped {
        return Err(e);
    }
    let problem = problem.map_err(HullQueryRefusal::Original)?;
    let scope = scope(
        &problem,
        &w2,
        &b2,
        plan.o,
        &objective,
        data.objective_constant,
        check,
    )?;
    check()?;
    let claim = lease.retain().map_err(HullQueryRefusal::Resource)?;
    context.checkpoint(HullLeaseClass::Optional)?;
    Ok(HullOriginalOwner(Arc::new(OriginalOwner {
        problem,
        w2,
        b2,
        output_dim: plan.o,
        objective,
        original_constant: data.objective_constant,
        scope,
        context: context.clone(),
        _claim: claim,
    })))
}

fn scope(
    problem: &HullMarginProblem,
    w2: &[f64],
    b2: &[f64],
    output: usize,
    objective: &[(usize, f64)],
    constant: f64,
    mut checkpoint: impl FnMut() -> Result<(), HullQueryRefusal>,
) -> Result<[u8; 32], HullQueryRefusal> {
    checkpoint()?;
    let mut hash = Sha256::new();
    hash.update(b"ny.hull.original.binary32-weights.exact-bias.wholebox.v1\0");
    for n in [
        problem.input_dim,
        problem.hidden_dim,
        output,
        objective.len(),
    ] {
        hash.update(work(n)?.to_le_bytes());
    }
    hash.update(constant.to_bits().to_le_bytes());
    for values in [
        &problem.w1[..],
        &problem.b1[..],
        w2,
        b2,
        &problem.lower[..],
        &problem.upper[..],
    ] {
        for (i, v) in values.iter().enumerate() {
            if i % 1024 == 0 {
                checkpoint()?;
            }
            hash.update(v.to_bits().to_le_bytes());
        }
    }
    for (i, &(index, value)) in objective.iter().enumerate() {
        if i % 1024 == 0 {
            checkpoint()?;
        }
        hash.update(work(index)?.to_le_bytes());
        hash.update(value.to_bits().to_le_bytes());
    }
    checkpoint()?;
    Ok(hash.finalize().into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HullProofOrigin {
    CheckedQuery,
    OriginalReplay,
}

struct CheckedProof {
    original: HullOriginalOwner,
    multipliers: Arc<HullMultipliers>,
    bound: f64,
    scope: [u8; 32],
    origin: HullProofOrigin,
    triangle_search: HullTriangleSearchStats,
    _claims: Vec<Arc<dyn HullRetainedResource>>,
    // A replay keeps the same admitted query evidence; clones never duplicate it.
    _query: Option<HullCheckedCertificate>,
}

/// Sealed mathematical whole-box certificate; never accepts report fields or
/// caller flags. It covers the owner's one complete original margin objective.
#[derive(Clone)]
pub struct HullCheckedCertificate(Arc<CheckedProof>);
impl HullCheckedCertificate {
    pub fn lower_bound(&self) -> f64 {
        self.0.bound
    }
    pub fn verified(&self) -> bool {
        self.0.bound > 0.0
    }
    pub fn scope(&self) -> [u8; 32] {
        self.0.scope
    }
    pub fn origin(&self) -> HullProofOrigin {
        self.0.origin
    }
    pub fn original(&self) -> HullOriginalOwner {
        self.0.original.clone()
    }
    /// Completed work of the originating query, preserved by fresh replay.
    /// The two search evaluations were advice, never certificate proposals.
    pub fn triangle_search(&self) -> HullTriangleSearchStats {
        self.0.triangle_search
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HullTriangleSearchStats {
    pub search_attempts: u64,
    /// Search invocations that completed with a finite directed score.
    pub search_evaluations: u64,
    pub certificate_proposals: u64,
    pub certificate_evaluations: u64,
    pub selected_policy: PositiveLowerSlopePolicy,
}

pub struct HullTriangleResult {
    /// Telemetry; mutating it does not modify the private certificate.
    pub report: HullRootReport,
    pub certificate: HullCheckedCertificate,
}
impl HullTriangleResult {
    /// Nonpositive/tied results need the unqualified LP/BaB profile. The
    /// existing public engines remain available, but gain no shared-resource
    /// claim from this bounded triangle ingress.
    pub fn require_strict_positive(&self) -> Result<(), HullQueryRefusal> {
        if self.certificate.verified() {
            Ok(())
        } else {
            Err(HullQueryRefusal::UnpricedLpOrBab)
        }
    }
}

fn same_context(
    owner: &HullOriginalOwner,
    context: &HullQueryContext,
) -> Result<(), HullQueryRefusal> {
    if owner.0.context.shares_authority(context) {
        Ok(())
    } else {
        Err(HullQueryRefusal::ForeignAuthority)
    }
}
fn certify(
    problem: &HullMarginProblem,
    multipliers: &HullMultipliers,
    lease: &dyn HullResourceLease,
    context: &HullQueryContext,
    class: HullLeaseClass,
) -> Result<f64, HullQueryRefusal> {
    let mut stopped = None;
    let mut check = || match poll(lease, context, class) {
        Ok(()) => true,
        Err(e) => {
            stopped = Some(e);
            false
        }
    };
    let result = certify_hull_lower_bound_polled(problem, multipliers, &mut check);
    if let Some(e) = stopped {
        return Err(e);
    }
    result.ok_or(HullQueryRefusal::Numerical)
}

/// Paid untrusted search followed by one finalized original proposal and the
/// unchanged original closed form. Exact score ties keep legacy multipliers.
/// No LP model or AY workspace is constructed by this profile.
pub fn checked_hull_triangle(
    owner: &HullOriginalOwner,
    context: &HullQueryContext,
) -> Result<HullTriangleResult, HullQueryRefusal> {
    same_context(owner, context)?;
    context.checkpoint(HullLeaseClass::Optional)?;
    let p = &owner.0.problem;
    let plan = GeometryPlan::new(
        p.input_dim,
        p.hidden_dim,
        owner.0.output_dim,
        owner.0.objective.len(),
    )?;
    let lease = context.lease(HullLeaseClass::Optional, plan.requests.triangle)?;
    let mut stopped = None;
    let mut check = || match poll(&*lease, context, HullLeaseClass::Optional) {
        Ok(()) => true,
        Err(e) => {
            stopped = Some(e);
            false
        }
    };
    let bounds = preactivation_bounds_polled(p, &mut check);
    if let Some(e) = stopped {
        return Err(e);
    }
    let bounds = bounds.ok_or(HullQueryRefusal::Numerical)?;
    poll(&*lease, context, HullLeaseClass::Optional)?;
    let roles = classify(p, &bounds);
    let neg = roles.iter().filter(|r| **r == Role::NegUnstable).count();
    let pos = roles.iter().filter(|r| **r == Role::PosUnstable).count();
    let mut stopped = None;
    let mut check = || match poll(&*lease, context, HullLeaseClass::Optional) {
        Ok(()) => true,
        Err(e) => {
            stopped = Some(e);
            false
        }
    };
    let multipliers = triangle_multipliers_polled(p, &roles, &bounds, &mut check);
    if let Some(e) = stopped {
        return Err(e);
    }
    let mut multipliers = multipliers.ok_or(HullQueryRefusal::Numerical)?;
    // Validate existing capacities and reserve the sole added buffer before
    // copying. Its actual allocator capacity is reconciled with the same paid
    // scratch grant; there is never a second HN multiplier array.
    let multiplier_capacity = add(
        mul(
            add(multipliers.alpha.capacity(), multipliers.g.capacity())?,
            8,
        )?,
        1024,
    )?;
    if multiplier_capacity > plan.requests.triangle.persistent_bytes {
        return Err(HullQueryRefusal::Resource(HullResourceRefusal::Memory));
    }
    let scratch_required = |alpha_capacity: usize| {
        add(
            add(
                add(
                    mul(bounds.capacity(), size_of::<(f64, f64)>())?,
                    mul(roles.capacity(), size_of::<Role>())?,
                )?,
                mul(alpha_capacity, size_of::<f64>())?,
            )?,
            add(mul(p.input_dim, 16)?, 4096)?,
        )
    };
    if scratch_required(p.hidden_dim)? > plan.requests.triangle.scratch_bytes {
        return Err(HullQueryRefusal::Resource(HullResourceRefusal::Memory));
    }
    poll(&*lease, context, HullLeaseClass::Optional)?;
    let mut legacy_alpha = Vec::new();
    legacy_alpha
        .try_reserve_exact(p.hidden_dim)
        .map_err(|_| HullQueryRefusal::Resource(HullResourceRefusal::Memory))?;
    if scratch_required(legacy_alpha.capacity())? > plan.requests.triangle.scratch_bytes {
        return Err(HullQueryRefusal::Resource(HullResourceRefusal::Memory));
    }
    for (j, &alpha) in multipliers.alpha.iter().enumerate() {
        if j % 1024 == 0 {
            poll(&*lease, context, HullLeaseClass::Optional)?;
        }
        legacy_alpha.push(alpha);
    }
    // These scores are private search advice. No CheckedProof exists until
    // the selected vector has completed its separate final checker pass.
    let mut search_evaluations = 0;
    let legacy_score = certify(p, &multipliers, &*lease, context, HullLeaseClass::Optional)?;
    search_evaluations += 1;
    for (j, role) in roles.iter().enumerate() {
        if j % 1024 == 0 {
            poll(&*lease, context, HullLeaseClass::Optional)?;
        }
        if *role == Role::PosUnstable {
            let (lo, hi) = bounds[j];
            // Match the policy constructor's refusal to use nonfinite slope
            // estimates. The checker remains valid for either actual vector.
            if (hi / (hi - lo)).is_finite() {
                multipliers.alpha[j] = if hi >= -lo { 1.0 } else { 0.0 };
            }
        }
    }
    let area_score = match certify(p, &multipliers, &*lease, context, HullLeaseClass::Optional) {
        Ok(score) => {
            search_evaluations += 1;
            Some(score)
        }
        // Invalid optional advice cannot discard a usable legacy score.
        // Resource and environment refusal still stop the whole publication.
        Err(HullQueryRefusal::Numerical) => None,
        Err(e) => return Err(e),
    };
    let (selected_policy, selected_score) = if area_score.is_some_and(|score| score > legacy_score)
    {
        (PositiveLowerSlopePolicy::Area, area_score.unwrap())
    } else {
        // Includes exact floating ties and either sign of zero. Restore the
        // stored bits, rather than recomputing the legacy proposal.
        for (j, &alpha) in legacy_alpha.iter().enumerate() {
            if j % 1024 == 0 {
                poll(&*lease, context, HullLeaseClass::Optional)?;
            }
            multipliers.alpha[j] = alpha;
        }
        (PositiveLowerSlopePolicy::Proportional, legacy_score)
    };
    let bound = certify(p, &multipliers, &*lease, context, HullLeaseClass::Optional)?;
    if bound.to_bits() != selected_score.to_bits() {
        return Err(HullQueryRefusal::ReplayMismatch);
    }
    let triangle_search = HullTriangleSearchStats {
        search_attempts: 2,
        search_evaluations,
        certificate_proposals: 1,
        certificate_evaluations: 1,
        selected_policy,
    };
    // Complete transfers only persistent bytes; actual scratch owners must
    // drop before that transfer, rather than waiting for function return.
    drop(bounds);
    drop(roles);
    drop(legacy_alpha);
    poll(&*lease, context, HullLeaseClass::Optional)?;
    let claim = lease.retain().map_err(HullQueryRefusal::Resource)?;
    context.checkpoint(HullLeaseClass::Optional)?;
    let certificate = HullCheckedCertificate(Arc::new(CheckedProof {
        original: owner.clone(),
        multipliers: Arc::new(multipliers),
        bound,
        scope: owner.scope(),
        origin: HullProofOrigin::CheckedQuery,
        triangle_search,
        _claims: vec![claim],
        _query: None,
    }));
    context.checkpoint(HullLeaseClass::Optional)?;
    Ok(HullTriangleResult {
        report: HullRootReport {
            unstable: neg + pos,
            neg_unstable: neg,
            pos_unstable: pos,
            lp_value: None,
            certified: Some(bound),
            route: Some(HullBoundRoute::Triangle),
            lp_status: if bound > 0.0 {
                HullLpStatus::SkippedByGoal
            } else {
                HullLpStatus::NotBuilt
            },
            subgradient_iterations: 0,
        },
        certificate,
    })
}

/// Independently rebuild the original reduced margin and replay the original
/// closed-form arithmetic with the sealed multipliers, on the SAME ledger.
pub fn replay_hull_original(
    certificate: &HullCheckedCertificate,
    context: &HullQueryContext,
) -> Result<HullCheckedCertificate, HullQueryRefusal> {
    let owner = &certificate.0.original;
    same_context(owner, context)?;
    context.checkpoint(HullLeaseClass::Replay)?;
    let p = &owner.0.problem;
    let plan = GeometryPlan::new(
        p.input_dim,
        p.hidden_dim,
        owner.0.output_dim,
        owner.0.objective.len(),
    )?;
    let lease = context.lease(HullLeaseClass::Replay, plan.requests.replay)?;
    let check = || poll(&*lease, context, HullLeaseClass::Replay);
    let actual_scope = scope(
        p,
        &owner.0.w2,
        &owner.0.b2,
        owner.0.output_dim,
        &owner.0.objective,
        owner.0.original_constant,
        check,
    )?;
    if actual_scope != certificate.scope() || actual_scope != owner.scope() {
        return Err(HullQueryRefusal::ReplayMismatch);
    }
    let mut stopped = None;
    let mut checkpoint = || match check() {
        Ok(()) => true,
        Err(e) => {
            stopped = Some(e);
            false
        }
    };
    let margin = form_margin_down(
        p.hidden_dim,
        &owner.0.w2,
        &owner.0.b2,
        &owner.0.objective,
        owner.0.original_constant,
        &mut checkpoint,
    );
    if let Some(e) = stopped {
        return Err(e);
    }
    let (c, d) = margin.map_err(HullQueryRefusal::Original)?;
    if d.to_bits() != p.d.to_bits() || c.iter().zip(&p.c).any(|(a, b)| a.to_bits() != b.to_bits()) {
        return Err(HullQueryRefusal::ReplayMismatch);
    }
    drop(c);
    let bound = certify(
        p,
        &certificate.0.multipliers,
        &*lease,
        context,
        HullLeaseClass::Replay,
    )?;
    if bound.to_bits() != certificate.lower_bound().to_bits() {
        return Err(HullQueryRefusal::ReplayMismatch);
    }
    check()?;
    let claim = lease.retain().map_err(HullQueryRefusal::Resource)?;
    context.checkpoint(HullLeaseClass::Replay)?;
    let result = HullCheckedCertificate(Arc::new(CheckedProof {
        original: owner.clone(),
        multipliers: certificate.0.multipliers.clone(),
        bound,
        scope: actual_scope,
        origin: HullProofOrigin::OriginalReplay,
        triangle_search: certificate.0.triangle_search,
        _claims: vec![claim],
        _query: Some(certificate.0._query.as_ref().unwrap_or(certificate).clone()),
    }));
    context.checkpoint(HullLeaseClass::Replay)?;
    Ok(result)
}

#[cfg(test)]
#[path = "single_hidden_hull_resource_tests.rs"]
mod tests;
