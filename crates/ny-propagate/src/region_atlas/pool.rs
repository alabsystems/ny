// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Query-owned bounded proof pool. Entries share a charged immutable proof;
//! eviction drops only a cache reference and never changes a cover ledger.

use super::replay::{checked_direction_digest, shape, validate_and_hash};
use super::*;
use crate::certify::work::{LeaseClass, WorkBudget, WorkRequest};

/// Variants cannot be populated with a caller-supplied score or boolean.
#[derive(Debug, Clone)]
pub enum AtlasEntry {
    GeometryConflict(ScopeGeometryConflict),
    ObjectiveExclusion(ObjectiveExclusion),
    RegionMargin(RegionMargin),
    Facet(ReusableFacet),
}

impl AtlasEntry {
    fn common(&self) -> &CommonProof {
        match self {
            Self::GeometryConflict(proof) => &proof.0.common,
            Self::ObjectiveExclusion(proof) => &proof.0 .0.common,
            Self::RegionMargin(proof) => &proof.0.common,
            Self::Facet(proof) => &proof.0.common,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AtlasStats {
    pub inserted: u64,
    pub evicted: u64,
    pub lookups: u64,
    pub hits: u64,
}

#[derive(Debug)]
struct Slot {
    entry: AtlasEntry,
    last_use: u64,
}

#[derive(Debug)]
pub struct RegionAtlas {
    slots: Vec<Slot>,
    capacity: usize,
    clock: u64,
    stats: AtlasStats,
    reservation: PersistentReservation,
}

impl RegionAtlas {
    /// Charge the whole bounded slot pool before allocating. Individual proof
    /// reservations remain charged independently, including external owners.
    pub fn new(capacity: usize, budget: &WorkBudget) -> Result<Self, AtlasRefusal> {
        if capacity == 0 || capacity > 4096 {
            return Err(AtlasRefusal::ResourceLimit);
        }
        let bytes = capacity
            .checked_mul(size_of::<Slot>())
            .and_then(|n| n.checked_add(1024))
            .ok_or(AtlasRefusal::Overflow)?;
        let lease = budget.lease_for(
            LeaseClass::Replay,
            WorkRequest {
                persistent_bytes: bytes,
                ..WorkRequest::default()
            },
        )?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(capacity)
            .map_err(|_| AtlasRefusal::Allocation)?;
        if slots
            .capacity()
            .checked_mul(size_of::<Slot>())
            .and_then(|n| n.checked_add(1024))
            .ok_or(AtlasRefusal::Overflow)?
            > bytes
        {
            return Err(AtlasRefusal::Allocation);
        }
        let reservation = lease.commit_persistent()?;
        Ok(Self {
            slots,
            capacity,
            clock: 0,
            stats: AtlasStats::default(),
            reservation,
        })
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    pub fn stats(&self) -> AtlasStats {
        self.stats
    }
    pub fn reserved_pool_bytes(&self) -> usize {
        self.reservation.bytes()
    }

    /// Sealed arithmetic evidence alone can enter the pool. Insertion does not
    /// authorize a prune, remove a literal, or alter an existing proof's scope.
    pub fn insert_replayed(&mut self, entry: AtlasEntry) {
        self.clock = self.clock.saturating_add(1);
        self.stats.inserted = self.stats.inserted.saturating_add(1);
        let slot = Slot {
            entry,
            last_use: self.clock,
        };
        if self.slots.len() < self.capacity {
            self.slots.push(slot);
        } else {
            let replace = self
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, slot)| slot.last_use)
                .map(|(index, _)| index)
                .unwrap_or(0);
            self.slots[replace] = slot;
            self.stats.evicted = self.stats.evicted.saturating_add(1);
        }
    }

    fn lookup(
        &mut self,
        geometry: AffineGeometry<'_>,
        options: ReplayOptions,
        budget: &WorkBudget,
        predicate: impl Fn(&AtlasEntry) -> bool,
    ) -> Result<Option<AtlasEntry>, AtlasRefusal> {
        let (dimensions, terms) = shape(geometry, options)?;
        let work = terms
            .checked_mul(4)
            .and_then(|n| n.checked_add(self.slots.len().checked_mul(dimensions)?))
            .and_then(|n| n.checked_add(dimensions))
            .ok_or(AtlasRefusal::Overflow)?;
        let mut lease = budget.lease_for(
            LeaseClass::Replay,
            WorkRequest {
                scratch_bytes: 1024,
                work: u64::try_from(work).map_err(|_| AtlasRefusal::Overflow)?,
                ..WorkRequest::default()
            },
        )?;
        self.stats.lookups = self.stats.lookups.saturating_add(1);
        let premises = validate_and_hash(geometry, dimensions, &lease)?;
        let mut selected = None;
        for (index, slot) in self.slots.iter().enumerate() {
            lease.checkpoint()?;
            if predicate(&slot.entry) && slot.entry.common().covers(geometry, premises)? {
                selected = Some(index);
                break;
            }
        }
        lease.complete()?;
        let result = selected.map(|index| {
            self.clock = self.clock.saturating_add(1);
            self.slots[index].last_use = self.clock;
            self.stats.hits = self.stats.hits.saturating_add(1);
            self.slots[index].entry.clone()
        });
        Ok(result)
    }

    /// Geometry-only evidence may cross objective queries, provided original
    /// premises, identities and exact region containment still match.
    pub fn match_antecedent(
        &mut self,
        geometry: AffineGeometry<'_>,
        options: ReplayOptions,
        budget: &WorkBudget,
    ) -> Result<Option<ScopeGeometryConflict>, AtlasRefusal> {
        Ok(
            match self.lookup(geometry, options, budget, |entry| {
                matches!(entry, AtlasEntry::GeometryConflict(_))
            })? {
                Some(AtlasEntry::GeometryConflict(proof)) => Some(proof),
                _ => None,
            },
        )
    }

    pub fn find_margin(
        &mut self,
        geometry: AffineGeometry<'_>,
        objective: AffineObjective<'_>,
        options: ReplayOptions,
        budget: &WorkBudget,
    ) -> Result<Option<RegionMargin>, AtlasRefusal> {
        let digest = checked_direction_digest(
            geometry,
            objective.coefficients,
            objective.constant,
            b"ny-atlas/objective-le-zero/v1",
            options,
            budget,
        )?;
        if geometry.scope.property() != digest {
            return Err(AtlasRefusal::PropertyMismatch);
        }
        Ok(
            match self.lookup(geometry, options, budget, |entry| match entry {
                AtlasEntry::RegionMargin(proof) => proof.0.objective == digest,
                AtlasEntry::ObjectiveExclusion(proof) => proof.0 .0.objective == digest,
                _ => false,
            })? {
                Some(AtlasEntry::RegionMargin(proof)) => Some(proof),
                Some(AtlasEntry::ObjectiveExclusion(proof)) => Some(proof.0),
                _ => None,
            },
        )
    }

    pub fn find_exclusion(
        &mut self,
        geometry: AffineGeometry<'_>,
        objective: AffineObjective<'_>,
        options: ReplayOptions,
        budget: &WorkBudget,
    ) -> Result<Option<ObjectiveExclusion>, AtlasRefusal> {
        Ok(self
            .find_margin(geometry, objective, options, budget)?
            .and_then(|margin| margin.exclusion()))
    }

    pub fn find_facet(
        &mut self,
        geometry: AffineGeometry<'_>,
        facet: AffineInequality<'_>,
        options: ReplayOptions,
        budget: &WorkBudget,
    ) -> Result<Option<ReusableFacet>, AtlasRefusal> {
        let digest = checked_direction_digest(
            geometry,
            facet.coefficients,
            facet.upper,
            b"ny-atlas/facet-le/v1",
            options,
            budget,
        )?;
        Ok(
            match self.lookup(
                geometry,
                options,
                budget,
                |entry| matches!(entry, AtlasEntry::Facet(proof) if proof.0.digest == digest),
            )? {
                Some(AtlasEntry::Facet(proof)) => Some(proof),
                _ => None,
            },
        )
    }
}
