// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One immutable original head allocation and its shared resident reservation.
//!
//! This owns moved coefficient vectors, not a caller-supplied digest or mutable
//! external memory. A retained proof keeps this owner alive independently of
//! its tree. Shared accounting never skips original arithmetic or replay hashes.

use super::vocab_tree::{checked_mul, default_budget, identity};
use super::{
    AffineHead, BudgetedWireExport, DecoderRefusal, HeadArithmetic, HiddenBox, Meter,
    NextTokenReport, PreparedModelVocabTree, PreparedVocabTree, VocabEvidence, VocabOptions,
};
use crate::certify::scope::ScopeIdentity;
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkRequest};
use std::sync::Arc;

#[derive(Debug)]
struct HeadInner {
    vocab_size: usize,
    hidden_size: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
    arithmetic: HeadArithmetic,
    identity: ScopeIdentity,
    identity_terms: u64,
    budget: WorkBudget,
    reservation: PersistentReservation,
}

/// Minted only by the explicit resident capability. The original storage and
/// construction ledger remain untouched; this holds one caller-admitted finite
/// query and its separately priced binding metadata.
#[derive(Debug)]
struct ResidentQueryBinding {
    budget: WorkBudget,
    _reservation: PersistentReservation,
}

/// Process-local immutable head ownership. Clones share original coefficient
/// storage and one reservation; neither deserialization nor a digest can mint
/// this object. Query certificates still require independently supplied boxes.
/// Work and deadlines belong to one bounded owner session: dropping a query
/// releases its bytes but does not refund work or renew that session. Persistent
/// serving across separately funded queries requires a separate resident/query
/// ledger adapter; this API does not reset limits for a retained model.
#[derive(Debug, Clone)]
pub struct SharedAffineHead(Arc<HeadInner>, Option<Arc<ResidentQueryBinding>>);

impl SharedAffineHead {
    /// Transfer vectors without copying a vocabulary matrix. Spare capacities,
    /// metadata and the Arc control block are charged before ownership creation.
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        weights: Vec<f64>,
        bias: Vec<f64>,
        arithmetic: HeadArithmetic,
        options: &VocabOptions,
    ) -> Result<Self, DecoderRefusal> {
        if vocab_size == 0
            || hidden_size == 0
            || weights.len() != checked_mul(vocab_size, hidden_size)?
            || bias.len() != vocab_size
        {
            return Err(DecoderRefusal::Shape("shared head"));
        }
        if vocab_size > options.max_vocab || hidden_size > options.max_hidden {
            return Err(DecoderRefusal::Resource("shared head dimension cap"));
        }
        let coefficients = weights
            .capacity()
            .checked_add(bias.capacity())
            .ok_or(DecoderRefusal::Resource("shared head capacity"))?;
        let bytes = checked_mul(coefficients, size_of::<f64>())?
            .checked_add(size_of::<HeadInner>())
            .and_then(|bytes| bytes.checked_add(2 * size_of::<usize>()))
            .and_then(|bytes| bytes.checked_add(256))
            .ok_or(DecoderRefusal::Resource("shared head size"))?;
        if bytes as u64 > options.max_persistent_bytes || 4096 > options.max_scratch_bytes {
            return Err(DecoderRefusal::Resource("shared head memory cap"));
        }
        let work = weights
            .len()
            .checked_add(bias.len())
            .ok_or(DecoderRefusal::Resource("shared head work"))? as u64;
        if work > options.max_terms {
            return Err(DecoderRefusal::Resource("shared head work cap"));
        }
        let budget = default_budget(options)?;
        let mut meter = Meter::with_budget_class(options, budget.clone(), LeaseClass::Replay)?;
        let lease = budget
            .lease_for(
                LeaseClass::Replay,
                WorkRequest {
                    scratch_bytes: 4096,
                    persistent_bytes: bytes,
                    work,
                    proposals: 1,
                },
            )
            .map_err(DecoderRefusal::Budget)?;
        let head = AffineHead {
            vocab_size,
            hidden_size,
            weights: &weights,
            bias: &bias,
            arithmetic,
        };
        let checked_identity = identity(head, &mut meter)?;
        meter.checkpoint()?;
        let reservation = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
        Ok(Self(
            Arc::new(HeadInner {
                vocab_size,
                hidden_size,
                weights,
                bias,
                arithmetic,
                identity: checked_identity,
                identity_terms: meter.terms(),
                budget,
                reservation,
            }),
            None,
        ))
    }

    pub fn as_head(&self) -> AffineHead<'_> {
        AffineHead {
            vocab_size: self.0.vocab_size,
            hidden_size: self.0.hidden_size,
            weights: &self.0.weights,
            bias: &self.0.bias,
            arithmetic: self.0.arithmetic,
        }
    }
    pub fn head_identity(&self) -> &ScopeIdentity {
        &self.0.identity
    }
    pub fn identity_terms(&self) -> u64 {
        self.0.identity_terms
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0.reservation.bytes()
    }
    /// Includes query-binding metadata for local request-cap checks. Public
    /// head-storage counters retain their original allocation-only meaning.
    fn total_owner_capacity(&self) -> Result<usize, DecoderRefusal> {
        self.reserved_bytes()
            .checked_add(
                self.1
                    .as_ref()
                    .map_or(0, |binding| binding._reservation.bytes()),
            )
            .ok_or(DecoderRefusal::Resource("resident owner capacity"))
    }

    /// Original coefficients may belong to a separate explicit construction
    /// ledger. Include that still-live capacity once in a query-local census;
    /// binding metadata already belongs to the active query's live receipt.
    pub(super) fn resident_bytes_outside_ledger(&self, active: &WorkBudget) -> usize {
        if self.0.budget.shares_ledger(active) {
            0
        } else {
            self.reserved_bytes()
        }
    }

    pub fn budget(&self) -> WorkBudget {
        self.1
            .as_ref()
            .map_or_else(|| self.0.budget.clone(), |binding| binding.budget.clone())
    }

    /// Private bridge for ResidentAffineHead, never a public session renewal.
    /// Both byte authorities must be the same actual object. The caller has
    /// already created the finite query ledger; this creates no work allowance.
    pub(super) fn bind_resident_query(&self, query: &WorkBudget) -> Result<Self, DecoderRefusal> {
        if self.1.is_some() || !self.0.budget.shares_memory_authority(query) {
            return Err(DecoderRefusal::Resource(
                "resident head memory authority mismatch",
            ));
        }
        query
            .checkpoint_for(LeaseClass::Replay)
            .map_err(DecoderRefusal::Budget)?;
        let bytes = size_of::<ResidentQueryBinding>()
            .checked_add(2 * size_of::<usize>())
            .and_then(|n| n.checked_add(256))
            .ok_or(DecoderRefusal::Resource("resident query binding capacity"))?;
        let lease = query
            .lease_for(
                LeaseClass::Replay,
                WorkRequest {
                    scratch_bytes: 0,
                    persistent_bytes: bytes,
                    work: 1024,
                    proposals: 1,
                },
            )
            .map_err(DecoderRefusal::Budget)?;
        let reservation = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
        Ok(Self(
            self.0.clone(),
            Some(Arc::new(ResidentQueryBinding {
                budget: query.clone(),
                _reservation: reservation,
            })),
        ))
    }

    /// A replacement options ledger cannot grant this head fresh allowances.
    pub(super) fn scoped_options(
        &self,
        options: &VocabOptions,
    ) -> Result<VocabOptions, DecoderRefusal> {
        if options
            .budget
            .as_ref()
            .is_some_and(|requested| !requested.shares_ledger(&self.budget()))
        {
            return Err(DecoderRefusal::Resource(
                "shared head caller ledger mismatch",
            ));
        }
        if self.0.vocab_size > options.max_vocab
            || self.0.hidden_size > options.max_hidden
            || self.total_owner_capacity()? as u64 > options.max_persistent_bytes
        {
            return Err(DecoderRefusal::Resource("shared head request cap"));
        }
        let mut scoped = options.clone();
        scoped.budget = Some(self.budget());
        Ok(scoped)
    }

    pub(super) fn validate_binding(&self, head: AffineHead<'_>) -> Result<(), DecoderRefusal> {
        let owned = self.as_head();
        if head.vocab_size != owned.vocab_size
            || head.hidden_size != owned.hidden_size
            || head.arithmetic != owned.arithmetic
            || head.weights.len() != owned.weights.len()
            || head.bias.len() != owned.bias.len()
            || head.weights.as_ptr() != owned.weights.as_ptr()
            || head.bias.as_ptr() != owned.bias.as_ptr()
        {
            return Err(DecoderRefusal::Evidence("shared original-head ownership"));
        }
        Ok(())
    }

    pub(super) fn validate_identity(&self, checked: &ScopeIdentity) -> Result<(), DecoderRefusal> {
        if checked != self.head_identity() {
            return Err(DecoderRefusal::Evidence("shared original-head identity"));
        }
        Ok(())
    }

    pub fn prepare_vocab_tree(
        &self,
        anchor: &[f64],
        target: usize,
        options: &VocabOptions,
    ) -> Result<PreparedVocabTree<'_>, DecoderRefusal> {
        super::vocab_tree::prepare_with_owner(
            self.as_head(),
            anchor,
            target,
            options,
            Some(self.clone()),
        )
    }

    pub fn prepare_model_vocab_tree(
        &self,
        options: &VocabOptions,
    ) -> Result<PreparedModelVocabTree<'_>, DecoderRefusal> {
        super::model_tree::prepare_with_owner(self.as_head(), options, Some(self.clone()))
    }

    pub fn verify_next_token(
        &self,
        hidden: HiddenBox<'_>,
        anchor: &[f64],
        target: usize,
        options: &VocabOptions,
    ) -> Result<NextTokenReport, DecoderRefusal> {
        self.prepare_vocab_tree(anchor, target, options)?
            .verify(hidden, options)
    }

    /// Cold full-vocabulary box support without building prepared group geometry.
    /// Every original row is checked and the resulting cover is independently
    /// replayed. The same private coefficient owner suppresses only its repeated
    /// resident charge; work, deadlines and evidence remain on its bounded ledger.
    pub fn verify_dense_next_token(
        &self,
        hidden: HiddenBox<'_>,
        anchor: &[f64],
        target: usize,
        options: &VocabOptions,
    ) -> Result<NextTokenReport, DecoderRefusal> {
        super::certify::verify_dense_with_owner(
            self.as_head(),
            hidden,
            anchor,
            target,
            options,
            Some(self.clone()),
        )
    }

    pub fn replay_next_token(
        &self,
        hidden: HiddenBox<'_>,
        anchor: &[f64],
        target: usize,
        evidence: &VocabEvidence,
        options: &VocabOptions,
    ) -> Result<NextTokenReport, DecoderRefusal> {
        super::certify::replay_with_owner(
            self.as_head(),
            hidden,
            anchor,
            target,
            evidence,
            options,
            Some(self.clone()),
        )
    }

    pub fn replay_next_token_json(
        &self,
        hidden: HiddenBox<'_>,
        anchor: &[f64],
        target: usize,
        json: &[u8],
        options: &VocabOptions,
    ) -> Result<NextTokenReport, DecoderRefusal> {
        super::wire::replay_json_with_owner(
            self.as_head(),
            hidden,
            anchor,
            target,
            json,
            options,
            Some(self.clone()),
        )
    }

    pub fn export_json(
        &self,
        evidence: &VocabEvidence,
        options: &VocabOptions,
    ) -> Result<BudgetedWireExport, DecoderRefusal> {
        evidence.export_json(&self.scoped_options(options)?)
    }
}

/// Only a private, pointer-bound owner can suppress a repeated original-head
/// charge. Borrowed compatibility inputs always retain their complete charge.
pub(super) fn incremental_head_bytes(
    head: AffineHead<'_>,
    owner: Option<&SharedAffineHead>,
) -> Result<usize, DecoderRefusal> {
    if let Some(owner) = owner {
        owner.validate_binding(head)?;
        Ok(0)
    } else {
        checked_mul(
            head.weights
                .len()
                .checked_add(head.bias.len())
                .ok_or(DecoderRefusal::Resource("head size"))?,
            8,
        )
    }
}

pub(super) fn resident_with_owner(
    incremental: usize,
    owner: Option<&SharedAffineHead>,
) -> Result<usize, DecoderRefusal> {
    incremental
        .checked_add(
            owner
                .map(SharedAffineHead::total_owner_capacity)
                .transpose()?
                .unwrap_or(0),
        )
        .ok_or(DecoderRefusal::Resource("shared resident size"))
}
