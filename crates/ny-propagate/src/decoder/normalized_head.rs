// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Checked composition of a mathematical real final normalization and head.
//!
//! The input is an independently supplied pre-normalization box. No upstream
//! trajectory, tokenization, deployed normalization kernel or floating-point
//! head contract is inferred. Original binary64 coefficients remain intervals;
//! no binary32 carrier is used. Every original competitor is checked directly.
//! Group trees are a later extension. This route shares one region geometry
//! across the complete original-row scan and can retain a canonical head owner.
//!
//! Conditional decisions may stop at a checked dense quotient or Cauchy upper
//! once it is strictly negative. Replay instead requires an upper no greater
//! than the finite original proposal. Tighter old Taylor proposals still run
//! Taylor when needed. Generic normalization carriers and the full support
//! minimum are unchanged; no whole-graph dominance follows from this strategy.
//! Serialized method tags remain untrusted construction hints. Fresh stages
//! and visits are recorded separately, including in the sealed checked receipt.

use super::{
    AffineHead, DecoderRefusal, HeadArithmetic, HiddenBox, NextTokenProvenance, SharedAffineHead,
    VocabOptions,
};
use crate::certify::scope::{
    BoxRegion, CertificateScope, Digest, NumericalSemantics, ScopeIdentity, SCOPE_SCHEMA_VERSION,
};
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRequest};
use crate::layers::normalization::radial::f64_support::{GeometryF64, SupportDemand, SupportError};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;
use std::time::Instant;

pub mod shared;
pub mod wire;
// Separate private group evidence; old row/wire contracts remain unchanged.
#[allow(dead_code)]
pub(crate) mod groups;

const SCHEMA: u32 = 1;
const CHUNK: usize = 256;

/// Mathematical real normalization, with the authored epsilon unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinalNormalizationKind {
    LayerNorm,
    RmsNorm,
}

/// Original binary64 parameters, not narrowed or inferred from a model name.
#[derive(Debug, Clone, Copy)]
pub struct FinalNormalization<'a> {
    pub kind: FinalNormalizationKind,
    pub gamma: &'a [f64],
    pub beta: Option<&'a [f64]>,
    pub epsilon: f64,
}

/// An explicit original shared ledger is mandatory. Limits narrow that ledger;
/// no preparation/query/replay helper constructs another allowance internally.
#[derive(Debug, Clone, Copy)]
pub struct NormalizedHeadOptions<'a> {
    pub budget: &'a WorkBudget,
    pub limits: &'a VocabOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizedHeadRefusal {
    /// A rounded normalization output contract is required before composing
    /// any floating-point head contract, including canceled real directions.
    UnsupportedHeadArithmetic(HeadArithmetic),
    Invalid(DecoderRefusal),
}
impl From<DecoderRefusal> for NormalizedHeadRefusal {
    fn from(value: DecoderRefusal) -> Self {
        Self::Invalid(value)
    }
}
impl std::fmt::Display for NormalizedHeadRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for NormalizedHeadRefusal {}
type Result<T> = std::result::Result<T, NormalizedHeadRefusal>;

/// Untrusted construction hint in serialized evidence. Queries name the
/// selected checked support; replay can authorize that upper with an earlier
/// checked stage and preserves the old hint. Actual executed stages are in
/// report stats and the sealed certificate's checked stats, never this tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizedSupportMethod {
    ProjectedDense,
    RadialTaylor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedRowEvidence {
    pub competitor: usize,
    pub upper: f64,
    /// A query's construction hint, preserved by replay for wire compatibility.
    /// It cannot select arithmetic, authorize an upper, or replace actual stats.
    pub method: NormalizedSupportMethod,
}

/// Plain proposals. Serialization/deserialization conveys no proof authority.
/// Raw serde has external admission; no bounded wire parser/export is claimed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedHeadEvidence {
    pub schema_version: u32,
    pub scope_digest: [u8; 32],
    pub target: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub normalization: FinalNormalizationKind,
    /// Canonical ascending original IDs, excluding exactly the target.
    pub rows: Vec<NormalizedRowEvidence>,
}

/// Observed original-data/coordinate visits, distinct from reserved work.
/// Counts describe scans, not floating-point operations, wall time or speed.
/// Byte/work reservation fields are predictions under the explicit profile;
/// arbitrary original slice backing capacities and physical RSS are excluded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct NormalizedHeadStats {
    pub original_head_visits: u64,
    pub normalization_parameter_visits: u64,
    pub region_visits: u64,
    pub kernel_coordinate_visits: u64,
    pub coverage_visits: u64,
    pub projected_zero_rows: usize,
    pub projected_dense_rows: usize,
    pub radial_taylor_rows: usize,
    /// Dense quotient already met the query/replay threshold; no norm/Taylor.
    pub dense_screened_rows: usize,
    /// Cauchy met that threshold after the dense quotient; no Taylor.
    pub cauchy_screened_rows: usize,
    /// Actual Taylor attempts, including an attempt yielding no finite bound.
    pub radial_taylor_attempted_rows: usize,
    pub reserved_work: u64,
    pub persistent_bytes: usize,
    pub scratch_bytes: usize,
}

/// Only this module's checked complete-cover finish can mint this value.
/// Its private scope/decision remains sealed if raw report evidence is mutated.
#[derive(Debug, Clone)]
pub struct CertifiedNormalizedNextToken {
    inner: Arc<CertifiedNormalizedNextTokenInner>,
}
#[derive(Debug)]
struct CertifiedNormalizedNextTokenInner {
    target: usize,
    competitors: usize,
    maximum_upper: Option<f64>,
    scope: CertificateScope,
    provenance: NextTokenProvenance,
    checked_stats: NormalizedHeadStats,
    reservation: PersistentReservation,
    head_owner: Option<SharedAffineHead>,
}
impl CertifiedNormalizedNextToken {
    pub fn target(&self) -> usize {
        self.inner.target
    }
    pub fn covered_competitors(&self) -> usize {
        self.inner.competitors
    }
    pub fn maximum_competitor_upper(&self) -> Option<f64> {
        self.inner.maximum_upper
    }
    pub fn scope(&self) -> &CertificateScope {
        &self.inner.scope
    }
    pub fn provenance(&self) -> NextTokenProvenance {
        self.inner.provenance
    }
    /// Immutable original execution receipt. Mutating raw evidence/report stats
    /// cannot replace these checked stage counts or their measured visits.
    pub fn checked_stats(&self) -> &NormalizedHeadStats {
        &self.inner.checked_stats
    }
    pub fn reserved_bytes(&self) -> usize {
        self.inner.reservation.bytes()
    }
    pub fn shared_head_bytes(&self) -> usize {
        self.inner
            .head_owner
            .as_ref()
            .map_or(0, SharedAffineHead::reserved_bytes)
    }
}
#[derive(Debug, Clone)]
pub enum NormalizedNextTokenOutcome {
    Certified(CertifiedNormalizedNextToken),
    Unknown,
}

#[derive(Debug)]
pub struct NormalizedHeadReport {
    pub outcome: NormalizedNextTokenOutcome,
    pub evidence: NormalizedHeadEvidence,
    pub unresolved: Vec<usize>,
    pub stats: NormalizedHeadStats,
    provenance: NextTokenProvenance,
    reservation: PersistentReservation,
    budget: WorkBudget,
    head_owner: Option<SharedAffineHead>,
}
impl NormalizedHeadReport {
    /// Origin follows the sealed decision even if the compatibility public
    /// outcome is replaced with another valid certificate by its caller.
    pub fn provenance(&self) -> NextTokenProvenance {
        match &self.outcome {
            NormalizedNextTokenOutcome::Certified(proof) => proof.provenance(),
            NormalizedNextTokenOutcome::Unknown => self.provenance,
        }
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
    pub fn shared_head_bytes(&self) -> usize {
        self.head_owner
            .as_ref()
            .map_or(0, SharedAffineHead::reserved_bytes)
    }
    /// Reconstruct all original arithmetic on the query's original ledger.
    /// The supplied limits may tighten admission, never refresh its budget.
    pub fn replay(
        &self,
        head: AffineHead<'_>,
        norm: FinalNormalization<'_>,
        pre: HiddenBox<'_>,
        target: usize,
        limits: &VocabOptions,
    ) -> Result<Self> {
        run(
            head,
            norm,
            pre,
            target,
            Some(&self.evidence),
            NormalizedHeadOptions {
                budget: &self.budget,
                limits,
            },
            LeaseClass::Replay,
            NextTokenProvenance::OriginalReplay,
            self.head_owner.clone(),
        )
    }
}

fn resource(why: &'static str) -> NormalizedHeadRefusal {
    DecoderRefusal::Resource(why).into()
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(resource("normalized head size overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(resource("normalized head size overflow"))
}
fn reserve<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| resource("normalized head allocation"))?;
    Ok(values)
}
fn checkpoint(lease: &WorkLease, limits: &VocabOptions) -> std::result::Result<(), DecoderRefusal> {
    lease.checkpoint().map_err(DecoderRefusal::Budget)?;
    if limits.deadline.is_some_and(|d| Instant::now() >= d)
        || ny_core::instance_budget::deadline().is_some_and(|d| Instant::now() >= d)
    {
        return Err(DecoderRefusal::Deadline);
    }
    Ok(())
}
fn visits(counter: &mut u64, count: usize) -> std::result::Result<(), DecoderRefusal> {
    *counter = counter
        .checked_add(count as u64)
        .ok_or(DecoderRefusal::Resource("normalized visit overflow"))?;
    Ok(())
}
fn kernel_error(error: SupportError<DecoderRefusal>) -> NormalizedHeadRefusal {
    match error {
        SupportError::Poll(error) => error.into(),
        SupportError::Arithmetic(why) => DecoderRefusal::NonFinite(why).into(),
    }
}

/// Conservative original-coordinate/hash/coverage work for one complete
/// normalized query or replay. It does not grant an allowance or price wire I/O.
pub fn normalized_work_bound(vocab: usize, hidden: usize) -> Result<u64> {
    let work = add(
        add(mul(mul(vocab, hidden)?, 32)?, mul(hidden, 128)?)?,
        add(mul(vocab, 16)?, 4096)?,
    )?;
    u64::try_from(work).map_err(|_| resource("normalized work overflow"))
}

#[cfg(test)]
fn shape(
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    options: NormalizedHeadOptions<'_>,
) -> Result<WorkRequest> {
    shape_with_owner(head, norm, pre, target, options, None)
}

fn shape_with_owner(
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    options: NormalizedHeadOptions<'_>,
    owner: Option<&SharedAffineHead>,
) -> Result<WorkRequest> {
    let limits = options.limits;
    if let Some(owner) = owner {
        owner.validate_binding(head)?;
        if !owner.budget().shares_ledger(options.budget) {
            return Err(DecoderRefusal::Evidence("normalized original owner ledger").into());
        }
    }
    if head.arithmetic != HeadArithmetic::ExactReal {
        return Err(NormalizedHeadRefusal::UnsupportedHeadArithmetic(
            head.arithmetic,
        ));
    }
    if let Some(substitute) = &limits.budget {
        if !options.budget.shares_ledger(substitute) {
            return Err(DecoderRefusal::Evidence("normalized ledger substitution").into());
        }
    }
    if head.vocab_size == 0
        || target >= head.vocab_size
        || head.hidden_size == 0
        || head.bias.len() != head.vocab_size
        || head.weights.len() != mul(head.vocab_size, head.hidden_size)?
        || pre.lo.len() != head.hidden_size
        || pre.hi.len() != head.hidden_size
        || norm.gamma.len() != head.hidden_size
        || norm.beta.is_some_and(|b| b.len() != head.hidden_size)
    {
        return Err(DecoderRefusal::Shape("normalized original head/parameters/region").into());
    }
    if !norm.epsilon.is_finite()
        || norm.epsilon < crate::layers::normalization::NORMALIZATION_MIN_EPS as f64
    {
        return Err(DecoderRefusal::Shape("normalized unchanged epsilon domain").into());
    }
    if head.vocab_size > limits.max_vocab
        || head.hidden_size > limits.max_hidden
        || head.hidden_size > 65_536
        || head.vocab_size - 1 > limits.max_nodes
    {
        return Err(resource("normalized head dimensions"));
    }
    if limits.deadline.is_some_and(|d| Instant::now() >= d)
        || ny_core::instance_budget::deadline().is_some_and(|d| Instant::now() >= d)
    {
        return Err(DecoderRefusal::Deadline.into());
    }
    let evidence = add(
        mul(head.vocab_size - 1, size_of::<NormalizedRowEvidence>())?,
        size_of::<NormalizedHeadEvidence>(),
    )?;
    if evidence > limits.max_evidence_bytes {
        return Err(resource("normalized evidence cap"));
    }
    // Borrowed compatibility conservatively charges the complete logical
    // original head/parameter slices throughout the report lifetime. Arbitrary
    // external backing capacities remain the caller's responsibility; this is
    // no physical/RSS census. Only a validated canonical owner discounts its
    // already-reserved logical head bytes; owner=None receives no discount.
    // Owned scope copies,
    // evidence and worst unresolved IDs coexist within this component lease.
    let originals = mul(
        add(
            add(head.weights.len(), head.bias.len())?,
            add(norm.gamma.len(), norm.beta.map_or(0, |b| b.len()))?,
        )?,
        8,
    )?;
    let owner_discount = if owner.is_some() {
        mul(add(head.weights.len(), head.bias.len())?, 8)?
    } else {
        0
    };
    let originals = originals
        .checked_sub(owner_discount)
        .ok_or(resource("normalized owner discount"))?;
    let persistent = add(
        add(add(originals, evidence)?, mul(head.vocab_size, 8)?)?,
        add(mul(head.hidden_size, 16)?, 4096)?,
    )?;
    let scratch = add(mul(head.hidden_size, 512)?, 16 * 1024)?;
    // Units are original-coordinate, buffer/hash and coverage visits. The
    // f64 adapter uses at most 16 scans/row; completed stages and coordinate
    // visits are reported separately, while this reservation conservatively
    // retains all Taylor work even when a decision is screened earlier. 32
    // additionally covers original
    // head hashing, replay evidence checks and phase/scope scans.
    let work = normalized_work_bound(head.vocab_size, head.hidden_size)?;
    let request = WorkRequest {
        persistent_bytes: persistent,
        scratch_bytes: scratch,
        work,
        proposals: 1,
    };
    let live = options.budget.receipt().map_err(DecoderRefusal::Budget)?;
    let outside = owner.map_or(0, |owner| {
        owner.resident_bytes_outside_ledger(options.budget)
    });
    let total_resident = add(
        add(request.persistent_bytes, live.live_persistent_bytes)?,
        outside,
    )?;
    let total_scratch = add(request.scratch_bytes, live.live_scratch_bytes)?;
    if total_resident as u64 > limits.max_persistent_bytes
        || total_scratch as u64 > limits.max_scratch_bytes
        || request.work > limits.max_terms
        || limits.max_proposals == 0
    {
        return Err(resource("normalized resource prediction"));
    }
    Ok(request)
}

fn scope(
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    lease: &WorkLease,
    limits: &VocabOptions,
    stats: &mut NormalizedHeadStats,
    owner: Option<&SharedAffineHead>,
) -> Result<(CertificateScope, [u8; 32])> {
    let mut original = Sha256::new();
    original.update(b"ny-normalized-original-head/v1");
    original.update((head.vocab_size as u64).to_le_bytes());
    original.update((head.hidden_size as u64).to_le_bytes());
    let mut owner_hash = Sha256::new();
    owner_hash.update(b"ny-affine-vocabulary-head/v1");
    owner_hash.update((head.vocab_size as u64).to_le_bytes());
    owner_hash.update((head.hidden_size as u64).to_le_bytes());
    for values in [head.weights, head.bias] {
        for block in values.chunks(CHUNK) {
            checkpoint(lease, limits)?;
            visits(&mut stats.original_head_visits, block.len())?;
            for &value in block {
                if !value.is_finite() {
                    return Err(DecoderRefusal::NonFinite("normalized original head").into());
                }
                original.update(value.to_bits().to_le_bytes());
                if owner.is_some() {
                    owner_hash.update(value.to_bits().to_le_bytes());
                }
            }
        }
    }
    if let Some(owner) = owner {
        if Digest(owner_hash.finalize().into()) != owner.head_identity().model {
            return Err(DecoderRefusal::Evidence("normalized original owner identity").into());
        }
    }
    let head_digest = Digest(original.finalize().into());
    let mut parameters = Sha256::new();
    parameters.update(b"ny-mathematical-final-normalization/v1");
    parameters.update([if norm.kind == FinalNormalizationKind::LayerNorm {
        0
    } else {
        1
    }]);
    parameters.update(norm.epsilon.to_bits().to_le_bytes());
    parameters.update([u8::from(norm.beta.is_some())]);
    for values in [norm.gamma, norm.beta.unwrap_or(&[])] {
        for block in values.chunks(CHUNK) {
            checkpoint(lease, limits)?;
            visits(&mut stats.normalization_parameter_visits, block.len())?;
            for &value in block {
                if !value.is_finite() {
                    return Err(DecoderRefusal::NonFinite("normalized original parameters").into());
                }
                parameters.update(value.to_bits().to_le_bytes());
            }
        }
    }
    let norm_digest = Digest(parameters.finalize().into());
    let mut model = Sha256::new();
    model.update(b"ny-normalized-composition/v1");
    model.update(head_digest.0);
    model.update(norm_digest.0);
    let identity = ScopeIdentity {
        schema_version: SCOPE_SCHEMA_VERSION, model: Digest(model.finalize().into()),
        lowering: norm_digest,
        coordinates: Digest::of_bytes(b"pre-normalization-coordinates", &(head.hidden_size as u64).to_le_bytes()),
        assumptions: Digest::of_bytes(b"normalized-assumptions", b"caller-established-pre-normalization-box;mathematical-real-final-normalization;exact-real-original-head;no-upstream-or-runtime-claim/v1"),
        numerical: NumericalSemantics::ExactReal,
    };
    let property = Digest::of_bytes(b"strict-normalized-argmax", &(target as u64).to_le_bytes());
    let mut scope_hash = Sha256::new();
    scope_hash.update(b"ny-normalized-head-scope/v1");
    scope_hash.update(identity.model.0);
    scope_hash.update(property.0);
    let mut lower = reserve(head.hidden_size)?;
    let mut upper = reserve(head.hidden_size)?;
    for start in (0..head.hidden_size).step_by(CHUNK) {
        let end = (start + CHUNK).min(head.hidden_size);
        checkpoint(lease, limits)?;
        visits(&mut stats.region_visits, end - start)?;
        for i in start..end {
            let (l, u) = (pre.lo[i], pre.hi[i]);
            if !l.is_finite() || !u.is_finite() || l > u {
                return Err(DecoderRefusal::NonFinite("normalized ordered pre-region").into());
            }
            scope_hash.update(l.to_bits().to_le_bytes());
            scope_hash.update(u.to_bits().to_le_bytes());
            lower.push(l);
            upper.push(u);
        }
    }
    // E1 scope constructors revalidate endpoints without a callback. This
    // first profile hard-caps H at65536; checkpoints bracket those bounded
    // validators as well as the chunked copies/hash above.
    checkpoint(lease, limits)?;
    let region = BoxRegion::new(lower, upper)
        .map_err(|_| DecoderRefusal::Evidence("normalized scope region"))?;
    let scope = CertificateScope::new(identity, region, property)
        .map_err(|_| DecoderRefusal::Evidence("normalized scope"))?;
    checkpoint(lease, limits)?;
    Ok((scope, scope_hash.finalize().into()))
}

/// Check every original competitor directly, sharing only the region geometry.
pub fn verify_normalized_next_token(
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    options: NormalizedHeadOptions<'_>,
) -> Result<NormalizedHeadReport> {
    run(
        head,
        norm,
        pre,
        target,
        None,
        options,
        LeaseClass::Optional,
        NextTokenProvenance::CheckedQuery,
        None,
    )
}

/// Take the minimum of all available checked support candidates for every
/// original competitor.
/// Unlike the decision query, a negative dense or Cauchy bound does not stop
/// the remaining checked stages. This is useful when retaining tighter margins
/// for later replay. It uses the same worst-case work reservation, original
/// scope, arithmetic restrictions and complete-cover certificate validator.
pub fn verify_normalized_next_token_full_support(
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    options: NormalizedHeadOptions<'_>,
) -> Result<NormalizedHeadReport> {
    run_with_support(
        head,
        norm,
        pre,
        target,
        None,
        options,
        LeaseClass::Optional,
        NextTokenProvenance::CheckedQuery,
        None,
        true,
    )
}

/// Rehash parameters/box/head and reconstruct every original directional bound.
/// Claimed projections, methods and upper bounds never mint proof by themselves.
pub fn replay_normalized_next_token(
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    evidence: &NormalizedHeadEvidence,
    options: NormalizedHeadOptions<'_>,
) -> Result<NormalizedHeadReport> {
    run(
        head,
        norm,
        pre,
        target,
        Some(evidence),
        options,
        LeaseClass::Replay,
        NextTokenProvenance::OriginalReplay,
        None,
    )
}

fn run(
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    proposal: Option<&NormalizedHeadEvidence>,
    options: NormalizedHeadOptions<'_>,
    class: LeaseClass,
    provenance: NextTokenProvenance,
    head_owner: Option<SharedAffineHead>,
) -> Result<NormalizedHeadReport> {
    run_with_support(
        head, norm, pre, target, proposal, options, class, provenance, head_owner, false,
    )
}

fn run_with_support(
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    proposal: Option<&NormalizedHeadEvidence>,
    options: NormalizedHeadOptions<'_>,
    class: LeaseClass,
    provenance: NextTokenProvenance,
    head_owner: Option<SharedAffineHead>,
    full_support: bool,
) -> Result<NormalizedHeadReport> {
    ny_core::require_f64_interval_proof_environment()
        .map_err(|_| DecoderRefusal::Evidence("normalized binary64 proof environment"))?;
    let request = shape_with_owner(head, norm, pre, target, options, head_owner.as_ref())?;
    if let Some(evidence) = proposal {
        if evidence.schema_version != SCHEMA
            || evidence.target != target
            || evidence.vocab_size != head.vocab_size
            || evidence.hidden_size != head.hidden_size
            || evidence.normalization != norm.kind
            || evidence.rows.len() != head.vocab_size - 1
        {
            return Err(DecoderRefusal::Evidence("normalized replay shape binding").into());
        }
        if add(
            mul(evidence.rows.capacity(), size_of::<NormalizedRowEvidence>())?,
            size_of::<NormalizedHeadEvidence>(),
        )? > options.limits.max_evidence_bytes
        {
            return Err(resource("normalized proposal capacity"));
        }
    }
    let lease = options
        .budget
        .lease_for(class, request)
        .map_err(DecoderRefusal::Budget)?;
    let mut stats = NormalizedHeadStats {
        reserved_work: request.work,
        persistent_bytes: request.persistent_bytes,
        scratch_bytes: request.scratch_bytes,
        ..NormalizedHeadStats::default()
    };
    let (scope, digest) = scope(
        head,
        norm,
        pre,
        target,
        &lease,
        options.limits,
        &mut stats,
        head_owner.as_ref(),
    )?;
    if proposal.is_some_and(|p| p.scope_digest != digest) {
        return Err(DecoderRefusal::Evidence("normalized replay identity").into());
    }
    let mut check = |count| {
        checkpoint(&lease, options.limits)?;
        visits(&mut stats.kernel_coordinate_visits, count)
    };
    let geometry = GeometryF64::new(
        pre.lo,
        pre.hi,
        norm.epsilon,
        norm.kind == FinalNormalizationKind::LayerNorm,
        &mut check,
    )
    .map_err(kernel_error)?;
    let mut rows = reserve(head.vocab_size - 1)?;
    let mut unresolved = reserve(head.vocab_size - 1)?;
    for competitor in 0..head.vocab_size {
        if competitor == target {
            continue;
        }
        // All original rows were already rehashed into the scope. A replay
        // still requires the next canonical ID and a finite proposal, before
        // asking the checked support engine to meet that exact upper.
        let claimed = proposal.map(|evidence| &evidence.rows[rows.len()]);
        if claimed.is_some_and(|row| row.competitor != competitor || !row.upper.is_finite()) {
            return Err(DecoderRefusal::Evidence("normalized original-row replay").into());
        }
        let demand = claimed.map_or(SupportDemand::StrictNegative, |row| {
            SupportDemand::AtMost(row.upper)
        });
        let mut check = |count| {
            checkpoint(&lease, options.limits)?;
            visits(&mut stats.kernel_coordinate_visits, count)
        };
        let bound = if full_support {
            geometry.upper(
                head.row(competitor),
                head.row(target),
                norm.gamma,
                norm.beta,
                head.bias[competitor],
                head.bias[target],
                &mut check,
            )
        } else {
            geometry.upper_for(
                head.row(competitor),
                head.row(target),
                norm.gamma,
                norm.beta,
                head.bias[competitor],
                head.bias[target],
                demand,
                &mut check,
            )
        }
        .map_err(kernel_error)?;
        let method = if bound.taylor_upper.is_some_and(|v| v < bound.dense_upper) {
            stats.radial_taylor_rows += 1;
            NormalizedSupportMethod::RadialTaylor
        } else {
            stats.projected_dense_rows += 1;
            NormalizedSupportMethod::ProjectedDense
        };
        stats.dense_screened_rows += usize::from(bound.dense_screened);
        stats.cauchy_screened_rows += usize::from(bound.cauchy_screened);
        stats.radial_taylor_attempted_rows += usize::from(bound.taylor_attempted);
        stats.projected_zero_rows += usize::from(bound.projection_zero);
        // Canonical coverage requires no vocabulary-sized bitmap. Every raw
        // replay ID must equal the next original ID and every row is rebuilt.
        let upper = if let Some(claimed) = claimed {
            if claimed.upper < bound.upper {
                return Err(DecoderRefusal::Evidence("normalized original-row replay").into());
            }
            claimed.upper
        } else {
            bound.upper
        };
        if upper >= 0.0 {
            unresolved.push(competitor);
        }
        rows.push(NormalizedRowEvidence {
            competitor,
            upper,
            // Preserve old wire proposals byte-for-byte. This tag is a raw
            // construction hint; the stats above record fresh actual stages.
            method: claimed.map_or(method, |row| row.method),
        });
    }
    drop(geometry);
    let evidence = NormalizedHeadEvidence {
        schema_version: SCHEMA,
        scope_digest: digest,
        target,
        vocab_size: head.vocab_size,
        hidden_size: head.hidden_size,
        normalization: norm.kind,
        rows,
    };
    finish(
        scope, evidence, unresolved, stats, lease, options, provenance, head_owner,
    )
}

fn finish(
    scope: CertificateScope,
    evidence: NormalizedHeadEvidence,
    unresolved: Vec<usize>,
    mut stats: NormalizedHeadStats,
    mut lease: WorkLease,
    options: NormalizedHeadOptions<'_>,
    provenance: NextTokenProvenance,
    head_owner: Option<SharedAffineHead>,
) -> Result<NormalizedHeadReport> {
    if evidence.rows.len() != evidence.vocab_size - 1 {
        return Err(DecoderRefusal::Evidence("normalized complete cover").into());
    }
    let mut maximum: Option<f64> = None;
    let mut row_index = 0usize;
    let mut open_index = 0usize;
    for competitor in 0..evidence.vocab_size {
        if competitor == evidence.target {
            continue;
        }
        if row_index.is_multiple_of(CHUNK) {
            checkpoint(&lease, options.limits)?;
        }
        let row = &evidence.rows[row_index];
        if row.competitor != competitor || !row.upper.is_finite() {
            return Err(DecoderRefusal::Evidence("normalized canonical complete cover").into());
        }
        if row.upper >= 0.0 {
            if unresolved.get(open_index) != Some(&competitor) {
                return Err(DecoderRefusal::Evidence("normalized unresolved cover").into());
            }
            open_index += 1;
        }
        maximum = Some(maximum.map_or(row.upper, |value| value.max(row.upper)));
        row_index += 1;
        visits(&mut stats.coverage_visits, 1)?;
    }
    if open_index != unresolved.len() {
        return Err(DecoderRefusal::Evidence("normalized extra unresolved").into());
    }
    if add(
        mul(evidence.rows.capacity(), size_of::<NormalizedRowEvidence>())?,
        size_of::<NormalizedHeadEvidence>(),
    )? > options.limits.max_evidence_bytes
    {
        return Err(resource("normalized actual evidence capacity"));
    }
    checkpoint(&lease, options.limits)?;
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let reservation = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
    let outcome = if maximum.is_none_or(|value| value < 0.0) {
        NormalizedNextTokenOutcome::Certified(CertifiedNormalizedNextToken {
            inner: Arc::new(CertifiedNormalizedNextTokenInner {
                target: evidence.target,
                competitors: evidence.vocab_size - 1,
                maximum_upper: maximum,
                scope,
                provenance,
                checked_stats: stats.clone(),
                reservation: reservation.clone(),
                head_owner: head_owner.clone(),
            }),
        })
    } else {
        NormalizedNextTokenOutcome::Unknown
    };
    // The original class deadline still applies after the private mint path.
    options
        .budget
        .checkpoint_for(if provenance == NextTokenProvenance::OriginalReplay {
            LeaseClass::Replay
        } else {
            LeaseClass::Optional
        })
        .map_err(DecoderRefusal::Budget)?;
    if options.limits.deadline.is_some_and(|d| Instant::now() >= d)
        || ny_core::instance_budget::deadline().is_some_and(|d| Instant::now() >= d)
    {
        return Err(DecoderRefusal::Deadline.into());
    }
    Ok(NormalizedHeadReport {
        outcome,
        evidence,
        unresolved,
        stats,
        provenance,
        reservation,
        budget: options.budget.clone(),
        head_owner,
    })
}

#[cfg(test)]
mod tests;

// One existing scope-owner allocation now retains the whole sealed header.
const _: () = assert!(
    size_of::<NormalizedHeadReport>()
        + size_of::<CertifiedNormalizedNextTokenInner>()
        + 2 * size_of::<usize>()
        <= 4096
);
