// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Promotion-grade retained-domain v2 authority layered over the v1 phase.
//!
//! V1 remains the full-upload compatibility surface. V2 adds an exact,
//! append-only zero-phase split history and core-owned logical resident slots.
//! Every retained f32 family and history buffer is immutable for the lifetime
//! of a slot. Backends must use separate working/output buffers; a mutated
//! retained buffer cannot later authorize `CopyParent` without a new typed,
//! core-known snapshot and receipt.
//!
//! Core validation is deliberately structural: it proves wire grammar,
//! ranges, exact parent prefixes, suffix/order/pattern identities, ownership,
//! and resource receipts. The qualified numerical TCB must decode the exact
//! versioned topology, map every history node ID to one unique execution-order
//! ReLU/Sign preactivation with the correct flattened width, bind each phase
//! bit to beta sign/order and all six operand slices for that same logical
//! domain/phase, and reject every semantic mismatch.

use super::*;

#[cfg(test)]
use std::{
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(test)]
static RESIDENT_SNAPSHOT_MATERIALIZATIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
thread_local! {
    static RESIDENT_ALLOCATION_FAIL_COUNTDOWN: Cell<Option<usize>> = const { Cell::new(None) };
    static RESIDENT_VALIDATION_DEADLINE_INJECTION: Cell<Option<ResidentValidationDeadlineInjection>> = const { Cell::new(None) };
    static RESIDENT_JOURNAL_PANIC_INJECTION: Cell<Option<ResidentJournalPanicInjection>> = const { Cell::new(None) };
    static RESIDENT_COMPLETED_SETTLEMENT_PROBE: Cell<Option<ResidentCompletedSettlementProbe>> = const { Cell::new(None) };
}

#[cfg(test)]
#[derive(Clone, Copy, Default)]
struct ResidentCompletedSettlementProbe {
    settled: bool,
    diagnostics: usize,
    diagnostics_before_settlement: usize,
}

#[cfg(test)]
struct ScopedResidentCompletedSettlementProbe {
    previous: Option<ResidentCompletedSettlementProbe>,
}

#[cfg(test)]
impl Drop for ScopedResidentCompletedSettlementProbe {
    fn drop(&mut self) {
        RESIDENT_COMPLETED_SETTLEMENT_PROBE.with(|probe| probe.set(self.previous));
    }
}

#[cfg(test)]
fn inject_resident_completed_settlement_probe() -> ScopedResidentCompletedSettlementProbe {
    let previous = RESIDENT_COMPLETED_SETTLEMENT_PROBE
        .with(|probe| probe.replace(Some(ResidentCompletedSettlementProbe::default())));
    ScopedResidentCompletedSettlementProbe { previous }
}

#[cfg(test)]
fn resident_completed_settlement_probe_counts() -> (usize, usize) {
    RESIDENT_COMPLETED_SETTLEMENT_PROBE.with(|probe| {
        let state = probe
            .get()
            .expect("resident completed settlement probe is active");
        (state.diagnostics, state.diagnostics_before_settlement)
    })
}

#[cfg(test)]
fn resident_completed_settlement_begin() {
    RESIDENT_COMPLETED_SETTLEMENT_PROBE.with(|probe| {
        if let Some(mut state) = probe.get() {
            state.settled = false;
            probe.set(Some(state));
        }
    });
}

#[cfg(not(test))]
fn resident_completed_settlement_begin() {}

#[cfg(test)]
fn resident_completed_settlement_mark() {
    RESIDENT_COMPLETED_SETTLEMENT_PROBE.with(|probe| {
        if let Some(mut state) = probe.get() {
            state.settled = true;
            probe.set(Some(state));
        }
    });
}

#[cfg(not(test))]
fn resident_completed_settlement_mark() {}

fn resident_completed_terminal_detail(message: &'static str) -> String {
    #[cfg(test)]
    RESIDENT_COMPLETED_SETTLEMENT_PROBE.with(|probe| {
        if let Some(mut state) = probe.get() {
            state.diagnostics += 1;
            state.diagnostics_before_settlement += usize::from(!state.settled);
            probe.set(Some(state));
        }
    });
    message.into()
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct ResidentValidationDeadlineInjection {
    label: &'static str,
    hits_before_expiry: usize,
}

#[cfg(test)]
struct ScopedResidentValidationDeadlineInjection {
    previous: Option<ResidentValidationDeadlineInjection>,
}

#[cfg(test)]
struct ScopedResidentAllocationFailureInjection {
    previous: Option<usize>,
}

#[cfg(test)]
impl Drop for ScopedResidentAllocationFailureInjection {
    fn drop(&mut self) {
        RESIDENT_ALLOCATION_FAIL_COUNTDOWN.with(|countdown| countdown.set(self.previous));
    }
}

#[cfg(test)]
fn inject_resident_allocation_failure(
    successful_reserves_before_failure: usize,
) -> ScopedResidentAllocationFailureInjection {
    let previous = RESIDENT_ALLOCATION_FAIL_COUNTDOWN
        .with(|countdown| countdown.replace(Some(successful_reserves_before_failure)));
    ScopedResidentAllocationFailureInjection { previous }
}

#[cfg(test)]
impl Drop for ScopedResidentValidationDeadlineInjection {
    fn drop(&mut self) {
        RESIDENT_VALIDATION_DEADLINE_INJECTION.with(|injection| {
            injection.set(self.previous);
        });
    }
}

#[cfg(test)]
fn inject_resident_validation_deadline(
    label: &'static str,
    hits_before_expiry: usize,
) -> ScopedResidentValidationDeadlineInjection {
    let previous = RESIDENT_VALIDATION_DEADLINE_INJECTION.with(|injection| {
        injection.replace(Some(ResidentValidationDeadlineInjection {
            label,
            hits_before_expiry,
        }))
    });
    ScopedResidentValidationDeadlineInjection { previous }
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct ResidentJournalPanicInjection {
    label: &'static str,
    hits_before_panic: usize,
}

#[cfg(test)]
struct ScopedResidentJournalPanicInjection {
    previous: Option<ResidentJournalPanicInjection>,
}

#[cfg(test)]
impl Drop for ScopedResidentJournalPanicInjection {
    fn drop(&mut self) {
        RESIDENT_JOURNAL_PANIC_INJECTION.with(|injection| injection.set(self.previous));
    }
}

#[cfg(test)]
fn inject_resident_journal_panic(
    label: &'static str,
    hits_before_panic: usize,
) -> ScopedResidentJournalPanicInjection {
    let previous = RESIDENT_JOURNAL_PANIC_INJECTION.with(|injection| {
        injection.replace(Some(ResidentJournalPanicInjection {
            label,
            hits_before_panic,
        }))
    });
    ScopedResidentJournalPanicInjection { previous }
}

#[cfg(test)]
fn resident_maybe_inject_journal_panic(label: &str) {
    let panic_now = RESIDENT_JOURNAL_PANIC_INJECTION.with(|injection| match injection.get() {
        Some(state) if state.label == label && state.hits_before_panic == 0 => {
            injection.set(None);
            true
        }
        Some(mut state) if state.label == label => {
            state.hits_before_panic -= 1;
            injection.set(Some(state));
            false
        }
        _ => false,
    });
    assert!(
        !panic_now,
        "injected resident journal mutation panic at {label}"
    );
}

#[cfg(not(test))]
fn resident_maybe_inject_journal_panic(_label: &str) {}

fn resident_allocation_error(requested: usize, unit: &'static str) -> NyError {
    NyError::GpuBatchCapacityExceeded {
        requested,
        capacity: 0,
        unit,
        site: "gpu_bab_bound_resident_core_allocation",
    }
}

#[cfg(test)]
pub(super) fn resident_injected_allocation_failure() -> bool {
    RESIDENT_ALLOCATION_FAIL_COUNTDOWN.with(|countdown| match countdown.get() {
        Some(0) => {
            countdown.set(None);
            true
        }
        Some(remaining) => {
            countdown.set(Some(remaining - 1));
            false
        }
        None => false,
    })
}

#[cfg(test)]
pub(super) fn resident_injected_validation_deadline(label: &str) -> bool {
    RESIDENT_VALIDATION_DEADLINE_INJECTION.with(|injection| match injection.get() {
        Some(state) if state.label == label && state.hits_before_expiry == 0 => {
            injection.set(None);
            true
        }
        Some(mut state) if state.label == label => {
            state.hits_before_expiry -= 1;
            injection.set(Some(state));
            false
        }
        _ => false,
    })
}

#[cfg(not(test))]
pub(super) fn resident_injected_allocation_failure() -> bool {
    false
}

#[cfg(not(test))]
pub(super) fn resident_injected_validation_deadline(_label: &str) -> bool {
    false
}

fn resident_vec_with_capacity<T>(
    capacity: usize,
    unit: &'static str,
) -> std::result::Result<Vec<T>, GpuBabBoundResidentAdmissionError> {
    let mut values = Vec::new();
    if resident_injected_allocation_failure() {
        return Err(GpuBabBoundResidentAdmissionError::Allocation(
            resident_allocation_error(capacity, unit),
        ));
    }
    values.try_reserve_exact(capacity).map_err(|_| {
        GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(capacity, unit))
    })?;
    Ok(values)
}

fn resident_hash_set_with_capacity<T>(
    capacity: usize,
    unit: &'static str,
) -> std::result::Result<HashSet<T>, GpuBabBoundResidentAdmissionError>
where
    T: Eq + std::hash::Hash,
{
    let mut values = HashSet::new();
    if resident_injected_allocation_failure() {
        return Err(GpuBabBoundResidentAdmissionError::Allocation(
            resident_allocation_error(capacity, unit),
        ));
    }
    values.try_reserve(capacity).map_err(|_| {
        GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(capacity, unit))
    })?;
    Ok(values)
}

fn resident_vec_with_metadata_budget<T>(
    capacity: usize,
    charged_bytes_per_entry: usize,
    unit: &'static str,
    budget: &mut ResidentHostAdmissionBudget,
    deadline: ResidentValidationDeadline,
) -> std::result::Result<Vec<T>, GpuBabBoundResidentAdmissionError> {
    let values = resident_vec_with_capacity(capacity, unit)?;
    budget
        .charge_metadata_capacity(capacity, values.capacity(), charged_bytes_per_entry)
        .map_err(|()| {
            GpuBabBoundResidentAdmissionError::Decline(
                GpuBabBoundResidentWaveDecline::InsufficientCapacity,
            )
        })?;
    deadline
        .check("resident metadata vector reserve")
        .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
    Ok(values)
}

fn resident_hash_set_with_metadata_budget<T>(
    capacity: usize,
    charged_bytes_per_entry: usize,
    unit: &'static str,
    budget: &mut ResidentHostAdmissionBudget,
    deadline: ResidentValidationDeadline,
) -> std::result::Result<HashSet<T>, GpuBabBoundResidentAdmissionError>
where
    T: Eq + std::hash::Hash,
{
    let values = resident_hash_set_with_capacity(capacity, unit)?;
    budget
        .charge_metadata_capacity(capacity, values.capacity(), charged_bytes_per_entry)
        .map_err(|()| {
            GpuBabBoundResidentAdmissionError::Decline(
                GpuBabBoundResidentWaveDecline::InsufficientCapacity,
            )
        })?;
    deadline
        .check("resident metadata set reserve")
        .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
    Ok(values)
}

fn resident_copy_slice_with_charge<T: Copy>(
    source: &[T],
    unit: &'static str,
    budget: &mut ResidentHostAdmissionBudget,
    deadline: ResidentValidationDeadline,
) -> std::result::Result<Vec<T>, GpuBabBoundResidentAdmissionError> {
    let mut values = resident_vec_with_capacity(source.len(), unit)?;
    budget
        .charge_snapshot_capacity(source.len(), values.capacity(), size_of::<T>())
        .map_err(|()| {
            GpuBabBoundResidentAdmissionError::Decline(
                GpuBabBoundResidentWaveDecline::InsufficientCapacity,
            )
        })?;
    // Capacity was fallibly reserved and charged before the first copy; this
    // bounded extend cannot grow the allocation.
    for chunk in source.chunks(VALIDATION_POLL_STRIDE) {
        deadline
            .check("resident compact snapshot copy")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        values.extend_from_slice(chunk);
    }
    deadline
        .check("resident compact snapshot copy")
        .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
    Ok(values)
}

/// Maximum exact split-history words accepted across one candidate wave.
pub const GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS: usize = 1 << 26;
/// Maximum logical resident slots admitted by a v2 policy.
pub const GPU_BAB_BOUND_MAX_RESIDENT_DOMAIN_SLOTS: usize = 1 << 17;
/// Fixed words in one exact ReLU/Sign-at-zero history record.
pub const GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS: usize = 4;
/// Maximum appended ReLU/Sign decisions in one parent-to-children transition.
pub const GPU_BAB_BOUND_MAX_APPEND_SPLITS: usize = 16;
/// Absolute ceiling for retained-v2 core-host accountable charge.
///
/// This narrow v2 ledger covers configured slot reserve, compact v2 snapshots,
/// v2/maintenance pending and publication metadata, and base structural
/// scratch only when invoked by v2 admission. It does not cover generic v1
/// validation/results, caller/backend/result-owned memory, allocator RSS or
/// bookkeeping, or the one immediately inspected and dropped rejected reserve
/// described by [`GpuBabBoundResidentDomainPolicy`].
pub const GPU_BAB_BOUND_MAX_RETAINED_V2_CORE_HOST_CHARGED_BYTES: usize = if usize::BITS >= 64 {
    (14_u64 * 1024 * 1024 * 1024) as usize
} else {
    1_usize << 30
};
/// Absolute core ceiling for charged resident-domain payload in one phase.
pub const GPU_BAB_BOUND_MAX_RESIDENT_DEVICE_BYTES: usize = if usize::BITS >= 64 {
    (14_u64 * 1024 * 1024 * 1024) as usize
} else {
    1_usize << 30
};

const GPU_BAB_BOUND_SPLIT_HISTORY_TAG: u32 = 0x4E59_0100;
const GPU_BAB_BOUND_SPLIT_HISTORY_TAG_MASK: u32 = !1;
// Conservative requested-host-byte charges for core-owned headers, Vec/hash
// tables, accepted descriptors, tokens, and validation scratch. Payload bytes
// are charged separately and exactly. Every configured slot, including a
// vacant slot, permanently reserves this amount before the slot table is
// allocated. The reserve covers the slot-state allocation and the worst-case
// per-operation scratch used by zero-destination maintenance, so maintenance
// remains admissible when compact live payload has filled the rest of the host
// cap. It is deliberately much larger than the exact Rust struct footprints.
const GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES: usize = 2 << 10;
const GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES: usize = 1 << 12;
const GPU_BAB_BOUND_HOST_PENDING_GROUP_METADATA_BYTES: usize = 1 << 10;
const GPU_BAB_BOUND_HOST_PENDING_SOURCE_METADATA_BYTES: usize = 1 << 10;
pub(super) const GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES: usize = 1 << 6;
// Includes the complete fixed pending header (including its sealed completed
// memory template and zero-work receipt) while leaving the remainder of each
// configured 2-KiB slot reserve for the four operation-indexed containers.
const GPU_BAB_BOUND_HOST_MAINTENANCE_FIXED_BYTES_PER_OPERATION: usize = 9 << 7;

/// Monotone admission charge for workload-scaled core allocations.
///
/// The policy bounds admitted/live accountable charge, not allocator RSS. A
/// fallible reserve may momentarily return more capacity than requested; core
/// observes and charges that capacity while the new container is still empty.
/// If the charge would cross `limit`, the caller drops the rejected plan before
/// filling that container, attempting another allocation, entering raw
/// preflight, or publishing it into resident state.
#[derive(Debug)]
pub(super) struct ResidentHostAdmissionBudget {
    limit: usize,
    base_charged_bytes: usize,
    // The allocation-free prepass installs conservative charges for the
    // entire prospective plan up front. Each observed reserve below only adds
    // allocator over-capacity beyond its requested length, so the very first
    // oversized reserve is tested against every allocation still to come.
    metadata_charged_bytes: usize,
    snapshot_charged_bytes: usize,
}

impl ResidentHostAdmissionBudget {
    pub(super) fn new(
        limit: usize,
        base_charged_bytes: usize,
        nominal_metadata_charged_bytes: usize,
        nominal_snapshot_charged_bytes: usize,
    ) -> Result<Self> {
        if !matches!(
            base_charged_bytes
                .checked_add(nominal_metadata_charged_bytes)
                .and_then(|value| value.checked_add(nominal_snapshot_charged_bytes)),
            Some(total) if total <= limit
        ) {
            return Err(invalid(
                "resident prospective core-host charge exceeds the policy limit",
            ));
        }
        Ok(Self {
            limit,
            base_charged_bytes,
            metadata_charged_bytes: nominal_metadata_charged_bytes,
            snapshot_charged_bytes: nominal_snapshot_charged_bytes,
        })
    }

    fn checked_total_with(&self, metadata: usize, snapshot: usize) -> Option<usize> {
        self.base_charged_bytes
            .checked_add(metadata)
            .and_then(|value| value.checked_add(snapshot))
    }

    pub(super) fn replace_base_charge(&mut self, base_charged_bytes: usize) -> Result<()> {
        if !matches!(
            self.checked_total_with(self.metadata_charged_bytes, self.snapshot_charged_bytes)
                .and_then(|total| total.checked_sub(self.base_charged_bytes))
                .and_then(|non_base| non_base.checked_add(base_charged_bytes)),
            Some(total) if total <= self.limit
        ) {
            return Err(invalid(
                "resident observed slot-table charge exceeds the host budget",
            ));
        }
        self.base_charged_bytes = base_charged_bytes;
        Ok(())
    }

    pub(super) fn charge_metadata_capacity(
        &mut self,
        requested_capacity: usize,
        observed_capacity: usize,
        charged_bytes_per_entry: usize,
    ) -> std::result::Result<(), ()> {
        let excess_capacity = observed_capacity
            .checked_sub(requested_capacity)
            .ok_or(())?;
        let bytes = excess_capacity
            .checked_mul(charged_bytes_per_entry)
            .ok_or(())?;
        let metadata = self.metadata_charged_bytes.checked_add(bytes).ok_or(())?;
        if !matches!(
            self.checked_total_with(metadata, self.snapshot_charged_bytes),
            Some(total) if total <= self.limit
        ) {
            return Err(());
        }
        self.metadata_charged_bytes = metadata;
        Ok(())
    }

    fn charge_snapshot_capacity(
        &mut self,
        requested_capacity: usize,
        observed_capacity: usize,
        element_bytes: usize,
    ) -> std::result::Result<(), ()> {
        let excess_capacity = observed_capacity
            .checked_sub(requested_capacity)
            .ok_or(())?;
        let bytes = excess_capacity.checked_mul(element_bytes).ok_or(())?;
        let snapshot = self.snapshot_charged_bytes.checked_add(bytes).ok_or(())?;
        if !matches!(
            self.checked_total_with(self.metadata_charged_bytes, snapshot),
            Some(total) if total <= self.limit
        ) {
            return Err(());
        }
        self.snapshot_charged_bytes = snapshot;
        Ok(())
    }

    fn total_charged_bytes(&self) -> usize {
        self.checked_total_with(self.metadata_charged_bytes, self.snapshot_charged_bytes)
            .expect("admission budget updates checked total overflow")
    }
}

/// Exact owned dynamic u32 arena for append-only split histories.
///
/// This arena is structural evidence, not a topology proof. Numerical use is
/// authorized only after the qualified TCB performs the semantic
/// topology/history/beta/six-family association described by the module
/// contract.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuBabBoundSplitHistoryArena {
    words: GpuBabBoundOwnedSlice<u32>,
}

impl GpuBabBoundSplitHistoryArena {
    /// Construct a non-authoritative wire arena. Core validates every record,
    /// range, prefix, suffix, duplicate key, and schedule before acceptance.
    /// The producer must populate `words` through fallibly reserved capacity;
    /// construction moves its allocation without request-sized reallocation.
    #[must_use]
    pub fn new(words: Vec<u32>) -> Self {
        Self {
            words: GpuBabBoundOwnedSlice::new(words),
        }
    }

    #[must_use]
    pub fn words(&self) -> &[u32] {
        self.words.as_ref()
    }

    /// Observed backing-vector capacity used for narrow host accounting.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.words.capacity()
    }

    /// Conservative fixed header plus observed word-capacity bytes.
    #[must_use]
    pub fn accountable_bytes(&self) -> Option<usize> {
        self.words.accountable_bytes()
    }
}

/// Closed halfspace phase for one ReLU/Sign-at-zero split literal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundSplitHistoryPhase {
    /// Inactive branch, preactivation `z <= 0`.
    Inactive,
    /// Active branch, preactivation `z >= 0`.
    Active,
}

/// Typed producer form of one exact four-u32 split-history record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBabBoundSplitHistoryLiteral {
    pub phase: GpuBabBoundSplitHistoryPhase,
    pub topology_node_id: u32,
    pub neuron_index: u32,
    pub score: f32,
}

impl GpuBabBoundSplitHistoryLiteral {
    /// Encode `[0x4E590100 | phase, node, neuron, score.to_bits()]`.
    pub fn encode_words(self) -> Result<[u32; GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS]> {
        if self.topology_node_id == u32::MAX || !self.score.is_finite() {
            return Err(invalid(
                "typed split literal has a reserved node ID or nonfinite score",
            ));
        }
        let phase = match self.phase {
            GpuBabBoundSplitHistoryPhase::Inactive => 0,
            GpuBabBoundSplitHistoryPhase::Active => 1,
        };
        Ok([
            GPU_BAB_BOUND_SPLIT_HISTORY_TAG | phase,
            self.topology_node_id,
            self.neuron_index,
            self.score.to_bits(),
        ])
    }
}

/// Per-domain suffix view and stable truth-table pattern.
///
/// `branch_pattern` is independent of the compact surviving child ordinal.
/// Bit zero corresponds to the oldest record in this suffix.
/// For an `AppendReluChildren` group, views appear in canonical domain order
/// with strictly numerically ascending patterns. The admitted set may be a
/// strict subset of the truth table; omitted patterns convey no cover or prune
/// authority.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBabBoundSplitHistoryView {
    pub suffix: GpuBabBoundArenaRange,
    pub branch_pattern: u64,
}

/// Opaque one-owner reference to a core resident logical domain.
///
/// The fields are private and the type is intentionally neither `Copy` nor
/// `Clone`. Moving this value into a group source, release, or eviction is the
/// only safe-code route to consuming that exact slot generation.
#[derive(Debug, PartialEq, Eq)]
pub struct GpuBabBoundResidentSlotRef {
    session_nonce_sha256: [u8; 32],
    logical_domain_identity_sha256: [u8; 32],
    slot_index: u32,
    generation: u64,
}

impl GpuBabBoundResidentSlotRef {
    #[must_use]
    pub fn session_nonce_sha256(&self) -> &[u8; 32] {
        &self.session_nonce_sha256
    }

    #[must_use]
    pub fn logical_domain_identity_sha256(&self) -> &[u8; 32] {
        &self.logical_domain_identity_sha256
    }

    #[must_use]
    pub fn slot_index(&self) -> u32 {
        self.slot_index
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Read-only source/destination identity visible to the numerical TCB.
///
/// This audit transcript is not a capability and has no public constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuBabBoundResidentSlotTranscript {
    session_nonce_sha256: [u8; 32],
    logical_domain_identity_sha256: [u8; 32],
    slot_index: u32,
    generation: u64,
}

/// Core-owned physical expectation for one consumed logical source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundResidentSourcePresence {
    Resident,
    RefreshOnly,
}

/// Read-only source association presented to the numerical TCB.
///
/// A provider may observe an absent physical buffer only when this exact audit
/// says `RefreshOnly`; absence for `Resident` is terminal authority loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundResidentSourceAudit {
    transcript: GpuBabBoundResidentSlotTranscript,
    presence: GpuBabBoundResidentSourcePresence,
    family_payload_bytes: [usize; 6],
    history_payload_bytes: usize,
    resident_device_bytes: usize,
}

impl GpuBabBoundResidentSourceAudit {
    #[must_use]
    pub fn transcript(&self) -> GpuBabBoundResidentSlotTranscript {
        self.transcript
    }

    #[must_use]
    pub fn presence(&self) -> GpuBabBoundResidentSourcePresence {
        self.presence
    }

    #[must_use]
    pub fn family_payload_bytes(&self, family: GpuBabBoundResidentF32Family) -> usize {
        self.family_payload_bytes[family.index()]
    }

    #[must_use]
    pub fn history_payload_bytes(&self) -> usize {
        self.history_payload_bytes
    }

    #[must_use]
    pub fn resident_device_bytes(&self) -> usize {
        self.resident_device_bytes
    }
}

impl GpuBabBoundResidentSlotTranscript {
    #[must_use]
    pub fn session_nonce_sha256(&self) -> &[u8; 32] {
        &self.session_nonce_sha256
    }

    #[must_use]
    pub fn logical_domain_identity_sha256(&self) -> &[u8; 32] {
        &self.logical_domain_identity_sha256
    }

    #[must_use]
    pub fn slot_index(&self) -> u32 {
        self.slot_index
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl From<&GpuBabBoundResidentSlotRef> for GpuBabBoundResidentSlotTranscript {
    fn from(reference: &GpuBabBoundResidentSlotRef) -> Self {
        Self {
            session_nonce_sha256: reference.session_nonce_sha256,
            logical_domain_identity_sha256: reference.logical_domain_identity_sha256,
            slot_index: reference.slot_index,
            generation: reference.generation,
        }
    }
}

/// Exact source of one parent group's resident-domain construction.
#[derive(Debug, PartialEq, Eq)]
pub enum GpuBabBoundResidentParentSource {
    /// Upload every child family and its complete split history. `prior` is the
    /// explicit full-refresh/rehydration source retired only after Completed.
    FreshUpload {
        prior: Option<GpuBabBoundResidentSlotRef>,
    },
    /// Copy bit-identical parent families and prefix on device, uploading only
    /// whole changed families and the nonempty suffix.
    RetainedDelta { parent: GpuBabBoundResidentSlotRef },
}

impl GpuBabBoundResidentParentSource {
    fn token(&self) -> Option<&GpuBabBoundResidentSlotRef> {
        match self {
            Self::FreshUpload { prior } => prior.as_ref(),
            Self::RetainedDelta { parent } => Some(parent),
        }
    }

    fn is_delta(&self) -> bool {
        matches!(self, Self::RetainedDelta { .. })
    }
}

/// One existing parent group plus its shared exact history prefix and source.
#[derive(Debug, PartialEq, Eq)]
pub struct GpuBabBoundResidentParentGroup {
    pub parent_group_id: u64,
    pub prefix: GpuBabBoundArenaRange,
    pub construction: GpuBabBoundResidentConstruction,
    pub source: GpuBabBoundResidentParentSource,
}

/// Closed structural meaning of one parent group's child histories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundResidentConstruction {
    /// Append `1..=16` ordered ReLU/Sign-at-zero literals. In canonical domain
    /// order, sibling `branch_pattern` values are strictly numerically
    /// ascending. An adapter sorts whole child bundles and then rebuilds dense
    /// ordinals, ranges, and subchunks. Patterns need not cover the full truth
    /// table and convey no parent-cover or pruning authority.
    AppendReluChildren,
    /// Preserve the exact parent history. This covers explicit rehydration and
    /// input-box children whose box/f32 payload changes without a ReLU append.
    /// It conveys no sibling-cover or pruning authority.
    FreshReplace,
}

/// Owned v2 request. It is intentionally non-Clone because it owns slot refs.
///
/// `release` and `evict` are separate canonical schedule sections. Each must
/// be strictly ascending by `(slot_index, generation)`, they must be mutually
/// slot-disjoint, and both follow parent sources in the hashed schedule. A
/// producer sorts tokens before moving them into this request; core never
/// reorders capability ownership.
#[derive(Debug, PartialEq)]
pub struct GpuBabBoundResidentWaveRequest {
    wave: GpuBabBoundWaveRequest,
    split_history: GpuBabBoundSplitHistoryArena,
    parent_groups: Vec<GpuBabBoundResidentParentGroup>,
    domain_histories: Vec<GpuBabBoundSplitHistoryView>,
    release: Vec<GpuBabBoundResidentSlotRef>,
    evict: Vec<GpuBabBoundResidentSlotRef>,
}

impl GpuBabBoundResidentWaveRequest {
    #[must_use]
    pub fn new(
        wave: GpuBabBoundWaveRequest,
        split_history: GpuBabBoundSplitHistoryArena,
        parent_groups: Vec<GpuBabBoundResidentParentGroup>,
        domain_histories: Vec<GpuBabBoundSplitHistoryView>,
        release: Vec<GpuBabBoundResidentSlotRef>,
        evict: Vec<GpuBabBoundResidentSlotRef>,
    ) -> Self {
        Self {
            wave,
            split_history,
            parent_groups,
            domain_histories,
            release,
            evict,
        }
    }

    #[must_use]
    pub fn wave(&self) -> &GpuBabBoundWaveRequest {
        &self.wave
    }

    #[must_use]
    pub fn split_history(&self) -> &GpuBabBoundSplitHistoryArena {
        &self.split_history
    }

    #[must_use]
    pub fn parent_groups(&self) -> &[GpuBabBoundResidentParentGroup] {
        &self.parent_groups
    }

    #[must_use]
    /// Destination i's history corresponds exactly to `wave().domains[i]`
    /// in the request's canonical group-major domain order.
    pub fn domain_histories(&self) -> &[GpuBabBoundSplitHistoryView] {
        &self.domain_histories
    }

    #[must_use]
    pub fn release(&self) -> &[GpuBabBoundResidentSlotRef] {
        &self.release
    }

    #[must_use]
    pub fn evict(&self) -> &[GpuBabBoundResidentSlotRef] {
        &self.evict
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        GpuBabBoundWaveRequest,
        GpuBabBoundSplitHistoryArena,
        Vec<GpuBabBoundResidentParentGroup>,
        Vec<GpuBabBoundSplitHistoryView>,
        Vec<GpuBabBoundResidentSlotRef>,
        Vec<GpuBabBoundResidentSlotRef>,
    ) {
        (
            self.wave,
            self.split_history,
            self.parent_groups,
            self.domain_histories,
            self.release,
            self.evict,
        )
    }
}

/// Owned zero-destination release/eviction transaction.
///
/// Both token sections must be nonempty in aggregate, strictly ascending by
/// `(slot_index, generation)`, and disjoint. Every preaccept nonterminal returns
/// this exact non-Clone request.
#[derive(Debug, PartialEq, Eq)]
pub struct GpuBabBoundResidentMaintenanceRequest {
    release: Vec<GpuBabBoundResidentSlotRef>,
    evict: Vec<GpuBabBoundResidentSlotRef>,
    deadline: Instant,
}

impl GpuBabBoundResidentMaintenanceRequest {
    #[must_use]
    pub fn new(
        release: Vec<GpuBabBoundResidentSlotRef>,
        evict: Vec<GpuBabBoundResidentSlotRef>,
        deadline: Instant,
    ) -> Self {
        Self {
            release,
            evict,
            deadline,
        }
    }

    #[must_use]
    pub fn release(&self) -> &[GpuBabBoundResidentSlotRef] {
        &self.release
    }

    #[must_use]
    pub fn evict(&self) -> &[GpuBabBoundResidentSlotRef] {
        &self.evict
    }

    #[must_use]
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<GpuBabBoundResidentSlotRef>,
        Vec<GpuBabBoundResidentSlotRef>,
        Instant,
    ) {
        (self.release, self.evict, self.deadline)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SplitLiteralKey {
    node_id: u32,
    neuron: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitLiteralRecord {
    key: SplitLiteralKey,
    phase: u32,
    score_bits: u32,
}

fn decode_split_record(words: &[u32], label: &str, index: usize) -> Result<SplitLiteralRecord> {
    if words.len() != GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS {
        return Err(invalid(format!(
            "{label} record {index} does not contain exactly four u32 words"
        )));
    }
    let tag = words[0];
    if tag & GPU_BAB_BOUND_SPLIT_HISTORY_TAG_MASK != GPU_BAB_BOUND_SPLIT_HISTORY_TAG {
        return Err(invalid(format!(
            "{label} record {index} has an invalid or reserved split-history tag"
        )));
    }
    if words[1] == u32::MAX {
        return Err(invalid(format!(
            "{label} record {index} uses the reserved topology node ID"
        )));
    }
    if !f32::from_bits(words[3]).is_finite() {
        return Err(invalid(format!(
            "{label} record {index} has a nonfinite score"
        )));
    }
    Ok(SplitLiteralRecord {
        key: SplitLiteralKey {
            node_id: words[1],
            neuron: words[2],
        },
        phase: tag & 1,
        score_bits: words[3],
    })
}

fn split_words<'a>(
    range: GpuBabBoundArenaRange,
    arena: &'a [u32],
    label: &str,
) -> Result<&'a [u32]> {
    if !range
        .len
        .is_multiple_of(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
    {
        return Err(invalid(format!(
            "{label} word length must be a multiple of four"
        )));
    }
    let end = range.checked_end(arena.len(), label)?;
    Ok(&arena[range.start..end])
}

fn validate_literal_prefix(
    prefix: &[u32],
    label: &str,
    budget: &mut ResidentHostAdmissionBudget,
    deadline: ResidentValidationDeadline,
) -> std::result::Result<HashSet<SplitLiteralKey>, GpuBabBoundResidentAdmissionError> {
    let mut keys = resident_hash_set_with_metadata_budget(
        prefix.len() / GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS,
        GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
        "resident prefix literal keys",
        budget,
        deadline,
    )?;
    // `as_chunks::<N>()` (the tippy suggestion) reshapes this validation
    // walk's element type; keep `chunks_exact` until the public pin's clippy
    // also carries the lint and the rewrite can land for both toolchains.
    #[allow(unknown_lints)] // stock 1.95 clippy (public pin) does not know the lint below
    #[allow(clippy::chunks_exact_to_as_chunks)]
    for (record_index, words) in prefix
        .chunks_exact(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
        .enumerate()
    {
        deadline
            .poll(record_index, "resident split-history prefix validation")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let record = decode_split_record(words, label, record_index)?;
        if !keys.insert(record.key) {
            return Err(invalid_admission(format!(
                "{label} repeats a topology-node/neuron literal in its parent prefix"
            )));
        }
    }
    deadline
        .check("resident split-history prefix validation")
        .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
    Ok(keys)
}

fn validate_literal_suffix(
    prefix_keys: &HashSet<SplitLiteralKey>,
    suffix: &[u32],
    label: &str,
    budget: &mut ResidentHostAdmissionBudget,
    deadline: ResidentValidationDeadline,
) -> std::result::Result<(Vec<SplitLiteralRecord>, u64), GpuBabBoundResidentAdmissionError> {
    // The shared prefix is decoded and hashed exactly once per parent group.
    // A suffix is capped at sixteen records, so per-child validation work and
    // allocation cannot amplify with the (potentially long) prefix.
    let mut suffix_keys = resident_hash_set_with_metadata_budget(
        suffix.len() / GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS,
        GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
        "resident suffix literal keys",
        budget,
        deadline,
    )?;
    let mut suffix_records = resident_vec_with_metadata_budget(
        suffix.len() / GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS,
        GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
        "resident suffix literal records",
        budget,
        deadline,
    )?;
    let suffix_count = suffix.len() / GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS;
    if suffix_count > u64::BITS as usize {
        return Err(invalid_admission(format!(
            "{label} suffix exceeds the finite u64 branch-pattern width"
        )));
    }
    let mut branch_pattern = 0u64;
    // Same rationale as the prefix walk above: keep `chunks_exact` until the
    // public pin's clippy also carries the lint.
    #[allow(unknown_lints)] // stock 1.95 clippy (public pin) does not know the lint below
    #[allow(clippy::chunks_exact_to_as_chunks)]
    for (offset, words) in suffix
        .chunks_exact(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
        .enumerate()
    {
        deadline
            .poll(offset, "resident split-history suffix validation")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let record = decode_split_record(words, label, offset)?;
        if prefix_keys.contains(&record.key) || !suffix_keys.insert(record.key) {
            return Err(invalid_admission(format!(
                "{label} repeats a topology-node/neuron literal across prefix and suffix"
            )));
        }
        branch_pattern |= u64::from(record.phase) << offset;
        suffix_records.push(record);
    }
    deadline
        .check("resident split-history suffix validation")
        .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
    Ok((suffix_records, branch_pattern))
}

#[derive(Debug)]
struct GpuBabBoundResidentDomainSnapshot {
    activation: Vec<f32>,
    beta: Vec<f32>,
    abs: Vec<f32>,
    box_lower: Vec<f32>,
    box_upper: Vec<f32>,
    cached_la: Vec<f32>,
    history: Vec<u32>,
    logical_domain_identity_sha256: [u8; 32],
}

impl GpuBabBoundResidentDomainSnapshot {
    fn family(&self, family: GpuBabBoundResidentF32Family) -> &[f32] {
        self.family_vec(family).as_slice()
    }

    fn family_vec(&self, family: GpuBabBoundResidentF32Family) -> &Vec<f32> {
        match family {
            GpuBabBoundResidentF32Family::Activation => &self.activation,
            GpuBabBoundResidentF32Family::Beta => &self.beta,
            GpuBabBoundResidentF32Family::Abs => &self.abs,
            GpuBabBoundResidentF32Family::BoxLower => &self.box_lower,
            GpuBabBoundResidentF32Family::BoxUpper => &self.box_upper,
            GpuBabBoundResidentF32Family::CachedLa => &self.cached_la,
        }
    }
}

/// Closed retained f32 buffer families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundResidentF32Family {
    Activation,
    Beta,
    Abs,
    BoxLower,
    BoxUpper,
    CachedLa,
}

const RESIDENT_F32_FAMILIES: [GpuBabBoundResidentF32Family; 6] = [
    GpuBabBoundResidentF32Family::Activation,
    GpuBabBoundResidentF32Family::Beta,
    GpuBabBoundResidentF32Family::Abs,
    GpuBabBoundResidentF32Family::BoxLower,
    GpuBabBoundResidentF32Family::BoxUpper,
    GpuBabBoundResidentF32Family::CachedLa,
];

impl GpuBabBoundResidentF32Family {
    fn index(self) -> usize {
        match self {
            Self::Activation => 0,
            Self::Beta => 1,
            Self::Abs => 2,
            Self::BoxLower => 3,
            Self::BoxUpper => 4,
            Self::CachedLa => 5,
        }
    }
}

/// Finite retained-domain limits supplied by a reviewed backend session.
///
/// This v2 schema has zero explicit per-buffer padding: the accountable slot
/// charge is exactly six f32 payloads plus one u32 history payload. A backend
/// requiring an additional uploaded control buffer or explicit padding must
/// decline until a later schema accounts for that class.
///
/// `maximum_retained_v2_core_host_charged_bytes` narrowly covers the configured
/// slot reserve, compact v2 snapshots, v2/maintenance pending and publication
/// metadata, and the base structural scratch used inside v2 admission. It
/// excludes generic v1 validation/results, caller/backend/result-owned memory,
/// allocator RSS/overhead, and one rejected fallible reserve: core observes
/// that newly returned capacity while its container is empty and, if it would
/// exceed the complete prospective charge, immediately drops the plan before
/// filling it, attempting another allocation, entering raw preflight, or
/// publishing resident authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundResidentDomainPolicy {
    pub maximum_slots: usize,
    pub maximum_history_words: usize,
    pub maximum_retained_v2_core_host_charged_bytes: usize,
    pub maximum_resident_device_bytes: usize,
}

impl GpuBabBoundResidentDomainPolicy {
    #[must_use]
    pub fn is_valid(self) -> bool {
        let slot_table_reserve = self
            .maximum_slots
            .checked_mul(GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES);
        self.maximum_slots > 0
            && self.maximum_slots <= GPU_BAB_BOUND_MAX_RESIDENT_DOMAIN_SLOTS
            && self.maximum_history_words > 0
            && self.maximum_history_words <= GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS
            && self.maximum_retained_v2_core_host_charged_bytes > 0
            && self.maximum_retained_v2_core_host_charged_bytes
                <= GPU_BAB_BOUND_MAX_RETAINED_V2_CORE_HOST_CHARGED_BYTES
            && matches!(slot_table_reserve, Some(bytes) if bytes <= self.maximum_retained_v2_core_host_charged_bytes)
            && self.maximum_resident_device_bytes > 0
            && self.maximum_resident_device_bytes <= GPU_BAB_BOUND_MAX_RESIDENT_DEVICE_BYTES
    }
}

fn resident_policy_identity_sha256(policy: GpuBabBoundResidentDomainPolicy) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.resident-policy.v2\0");
    hash_u64(&mut hash, policy.maximum_slots as u64);
    hash_u64(&mut hash, policy.maximum_history_words as u64);
    hash_u64(
        &mut hash,
        policy.maximum_retained_v2_core_host_charged_bytes as u64,
    );
    hash_u64(&mut hash, policy.maximum_resident_device_bytes as u64);
    hash.finalize().into()
}

/// Bind one v1 request schedule to the exact latched resident-policy
/// observation. This keeps the full-upload compatibility path default-dark
/// while preventing a receipt prepared under Unsupported/one policy from
/// being replayed after the same session reports another policy.
pub(super) fn bind_v1_schedule_to_resident_policy(
    request_schedule_identity_sha256: [u8; 32],
    policy_observation_identity_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.v1-schedule-with-resident-policy.v2\0");
    hash.update(request_schedule_identity_sha256);
    hash.update(policy_observation_identity_sha256);
    hash.finalize().into()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GpuBabBoundResidentSlotLayout {
    family_payload_bytes: [usize; 6],
    history_payload_bytes: usize,
    payload_bytes: usize,
    core_host_charged_bytes: usize,
}

impl GpuBabBoundResidentSlotLayout {
    fn from_snapshot(snapshot: &GpuBabBoundResidentDomainSnapshot) -> Result<Self> {
        let mut family_payload_bytes = [0usize; 6];
        let mut payload_bytes = 0usize;
        let mut core_host_charged_bytes = 0usize;
        for family in RESIDENT_F32_FAMILIES {
            let values = snapshot.family_vec(family);
            let bytes = values
                .len()
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| invalid("resident f32 family bytes overflow usize"))?;
            family_payload_bytes[family.index()] = bytes;
            payload_bytes = payload_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("resident f32 payload total overflows usize"))?;
            core_host_charged_bytes = core_host_charged_bytes
                .checked_add(
                    values
                        .capacity()
                        .checked_mul(size_of::<f32>())
                        .ok_or_else(|| invalid("resident f32 host capacity overflows usize"))?,
                )
                .ok_or_else(|| invalid("resident host capacity total overflows usize"))?;
        }
        let history_payload_bytes = snapshot
            .history
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| invalid("resident history bytes overflow usize"))?;
        payload_bytes = payload_bytes
            .checked_add(history_payload_bytes)
            .ok_or_else(|| invalid("resident slot payload total overflows usize"))?;
        core_host_charged_bytes = core_host_charged_bytes
            .checked_add(
                snapshot
                    .history
                    .capacity()
                    .checked_mul(size_of::<u32>())
                    .ok_or_else(|| invalid("resident history host capacity overflows usize"))?,
            )
            .ok_or_else(|| invalid("resident host capacity total overflows usize"))?;
        Ok(Self {
            family_payload_bytes,
            history_payload_bytes,
            payload_bytes,
            core_host_charged_bytes,
        })
    }
}

/// Core-derived source of one accepted destination f32 family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundResidentFamilyTransfer {
    /// Full family H2D for a FreshUpload group.
    FreshUpload,
    /// Exact bit-identical family copied source-to-destination on device.
    CopyParent,
    /// Full changed family H2D for a RetainedDelta group.
    FreshReplace,
}

fn f32_bits_equal_with_deadline(
    left: &[f32],
    right: &[f32],
    deadline: ResidentValidationDeadline,
) -> Result<bool> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left
        .chunks(VALIDATION_POLL_STRIDE)
        .zip(right.chunks(VALIDATION_POLL_STRIDE))
    {
        deadline.check("resident family bit equality")?;
        if !left
            .iter()
            .zip(right)
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
        {
            return Ok(false);
        }
    }
    deadline.check("resident family bit equality")?;
    Ok(true)
}

#[derive(Debug)]
struct GpuBabBoundResidentDestinationPlan {
    slot_index: u32,
    generation: u64,
    base_domain_identity_sha256: [u8; 32],
    logical_domain_identity_sha256: [u8; 32],
    layout: GpuBabBoundResidentSlotLayout,
    source: Option<GpuBabBoundResidentSourceAudit>,
    family_transfers: [GpuBabBoundResidentFamilyTransfer; 6],
    history_prefix_bytes: usize,
    history_suffix_bytes: usize,
}

/// Read-only destination materialization plan visible after core acceptance.
#[derive(Debug)]
pub struct GpuBabBoundAcceptedResidentDomain {
    destination: GpuBabBoundResidentSlotTranscript,
    source: Option<GpuBabBoundResidentSourceAudit>,
    base_domain_identity_sha256: [u8; 32],
    logical_domain_identity_sha256: [u8; 32],
    family_transfers: [GpuBabBoundResidentFamilyTransfer; 6],
    family_payload_bytes: [usize; 6],
    history_prefix_bytes: usize,
    history_suffix_bytes: usize,
    resident_device_bytes: usize,
}

/// The same immutable descriptor is visible during pure preflight and after
/// acceptance; the alias emphasizes that no destination property is revealed
/// only after fallback authority is consumed.
pub type GpuBabBoundProposedResidentDomain = GpuBabBoundAcceptedResidentDomain;

impl GpuBabBoundAcceptedResidentDomain {
    #[must_use]
    pub fn destination(&self) -> GpuBabBoundResidentSlotTranscript {
        self.destination
    }

    #[must_use]
    pub fn source(&self) -> Option<GpuBabBoundResidentSourceAudit> {
        self.source
    }

    #[must_use]
    pub fn logical_domain_identity_sha256(&self) -> &[u8; 32] {
        &self.logical_domain_identity_sha256
    }

    #[must_use]
    pub fn base_domain_identity_sha256(&self) -> &[u8; 32] {
        &self.base_domain_identity_sha256
    }

    #[must_use]
    pub fn family_transfer(
        &self,
        family: GpuBabBoundResidentF32Family,
    ) -> GpuBabBoundResidentFamilyTransfer {
        self.family_transfers[family.index()]
    }

    #[must_use]
    pub fn family_payload_bytes(&self, family: GpuBabBoundResidentF32Family) -> usize {
        self.family_payload_bytes[family.index()]
    }

    #[must_use]
    pub fn history_prefix_bytes(&self) -> usize {
        self.history_prefix_bytes
    }

    #[must_use]
    pub fn history_suffix_bytes(&self) -> usize {
        self.history_suffix_bytes
    }

    #[must_use]
    pub fn resident_device_bytes(&self) -> usize {
        self.resident_device_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuBabBoundResidentPresence {
    Resident,
    RefreshOnly,
}

impl From<GpuBabBoundResidentPresence> for GpuBabBoundResidentSourcePresence {
    fn from(value: GpuBabBoundResidentPresence) -> Self {
        match value {
            GpuBabBoundResidentPresence::Resident => Self::Resident,
            GpuBabBoundResidentPresence::RefreshOnly => Self::RefreshOnly,
        }
    }
}

#[derive(Debug)]
struct GpuBabBoundResidentLiveSlot {
    generation: u64,
    snapshot: GpuBabBoundResidentDomainSnapshot,
    layout: GpuBabBoundResidentSlotLayout,
    presence: GpuBabBoundResidentPresence,
    in_flight: bool,
}

impl GpuBabBoundResidentLiveSlot {
    fn source_audit(
        &self,
        transcript: GpuBabBoundResidentSlotTranscript,
    ) -> GpuBabBoundResidentSourceAudit {
        GpuBabBoundResidentSourceAudit {
            transcript,
            presence: self.presence.into(),
            family_payload_bytes: self.layout.family_payload_bytes,
            history_payload_bytes: self.layout.history_payload_bytes,
            resident_device_bytes: if self.presence == GpuBabBoundResidentPresence::Resident {
                self.layout.payload_bytes
            } else {
                0
            },
        }
    }
}

#[derive(Debug)]
enum GpuBabBoundResidentSlotState {
    Vacant {
        high_generation: u64,
    },
    Live(GpuBabBoundResidentLiveSlot),
    Reserved {
        generation: u64,
        layout: GpuBabBoundResidentSlotLayout,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum GpuBabBoundResidentPolicyState {
    #[default]
    Unqueried,
    Unsupported,
    /// The exact session policy was observed and transcript-bound, but a v1
    /// call has not allocated the resident slot ledger.
    Observed(GpuBabBoundResidentDomainPolicy),
    Installed(GpuBabBoundResidentDomainPolicy),
}

#[derive(Debug, Default)]
pub(super) struct GpuBabBoundResidentDomainState {
    policy_state: GpuBabBoundResidentPolicyState,
    slots: Vec<GpuBabBoundResidentSlotState>,
    // Persistent duplicate-check scratch. It is installed and charged inside
    // the configured per-slot reserve, so pre-cap source validation performs
    // no heap/stack allocation and never needs new host headroom.
    source_authority_bitmap: Vec<u64>,
    completed_waves: u64,
    in_flight_slots: usize,
    reserved_slots: usize,
    poisoned: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct GpuBabBoundResidentLedgerAudit {
    pub(super) resident_device_bytes: usize,
    pub(super) core_host_charged_bytes: usize,
    pub(super) history_words: usize,
    pub(super) resident_slots: usize,
    pub(super) refresh_only_slots: usize,
    pub(super) in_flight_slots: usize,
    pub(super) reserved_slots: usize,
}

impl GpuBabBoundResidentDomainState {
    pub(super) fn poison_all(&mut self) {
        // Poison is an absorbing authority state, not an accounting erase.
        // Live and reserved layouts remain available so explicit close can
        // charge/release every still-owned residency lease. Generations stay
        // burned even when a terminal receipt was structurally invalid.
        self.poisoned = true;
    }

    pub(super) fn resources_are_quiescent(&self) -> bool {
        self.in_flight_slots == 0 && self.reserved_slots == 0
    }

    fn observe_supported_policy(&mut self, policy: GpuBabBoundResidentDomainPolicy) -> Result<()> {
        if !policy.is_valid() {
            return Err(invalid("retained-domain policy is malformed"));
        }
        match self.policy_state {
            GpuBabBoundResidentPolicyState::Observed(observed)
            | GpuBabBoundResidentPolicyState::Installed(observed)
                if observed != policy =>
            {
                self.poison_all();
                Err(invalid(
                    "retained-domain policy changed within one phase session",
                ))
            }
            GpuBabBoundResidentPolicyState::Observed(_)
            | GpuBabBoundResidentPolicyState::Installed(_) => Ok(()),
            GpuBabBoundResidentPolicyState::Unsupported => {
                self.poison_all();
                Err(invalid(
                    "retained-domain support changed from Unsupported to Installed",
                ))
            }
            GpuBabBoundResidentPolicyState::Unqueried => {
                self.policy_state = GpuBabBoundResidentPolicyState::Observed(policy);
                Ok(())
            }
        }
    }

    fn ensure_policy(
        &mut self,
        policy: GpuBabBoundResidentDomainPolicy,
        host_budget: Option<&mut ResidentHostAdmissionBudget>,
        deadline: Option<ResidentValidationDeadline>,
    ) -> std::result::Result<(), GpuBabBoundResidentAdmissionError> {
        self.observe_supported_policy(policy)
            .map_err(GpuBabBoundResidentAdmissionError::Poison)?;
        match self.policy_state {
            GpuBabBoundResidentPolicyState::Installed(_) => {
                if let Some(host_budget) = host_budget {
                    let actual_base_charge = self
                        .ledger_audit_with_deadline(deadline)
                        .map(|audit| audit.core_host_charged_bytes)
                        .map_err(|error| match error {
                            NyError::DeadlineExceeded(_) => {
                                GpuBabBoundResidentAdmissionError::Invalid(error)
                            }
                            _ => GpuBabBoundResidentAdmissionError::Poison(error),
                        })?;
                    host_budget
                        .replace_base_charge(actual_base_charge)
                        .map_err(|_| {
                            GpuBabBoundResidentAdmissionError::Allocation(
                                resident_allocation_error(
                                    self.slots.capacity(),
                                    "resident configured slot capacity charge",
                                ),
                            )
                        })?;
                }
                Ok(())
            }
            GpuBabBoundResidentPolicyState::Observed(observed) => {
                let mut slots = Vec::new();
                if resident_injected_allocation_failure() {
                    return Err(GpuBabBoundResidentAdmissionError::Allocation(
                        resident_allocation_error(
                            observed.maximum_slots,
                            "resident configured slot states",
                        ),
                    ));
                }
                slots
                    .try_reserve_exact(observed.maximum_slots)
                    .map_err(|_| {
                        GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                            observed.maximum_slots,
                            "resident configured slot states",
                        ))
                    })?;
                finish_resident_validation(deadline, "resident slot-table reserve")
                    .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                let observed_slot_table_charge = slots
                    .capacity()
                    .checked_mul(GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES)
                    .ok_or_else(|| {
                        GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                            observed.maximum_slots,
                            "resident configured slot capacity charge",
                        ))
                    })?;
                if observed_slot_table_charge > observed.maximum_retained_v2_core_host_charged_bytes
                {
                    return Err(GpuBabBoundResidentAdmissionError::Allocation(
                        resident_allocation_error(
                            observed.maximum_slots,
                            "resident configured slot capacity charge",
                        ),
                    ));
                }
                if let Some(host_budget) = host_budget {
                    host_budget
                        .replace_base_charge(observed_slot_table_charge)
                        .map_err(|_| {
                            GpuBabBoundResidentAdmissionError::Allocation(
                                resident_allocation_error(
                                    observed.maximum_slots,
                                    "resident configured slot capacity charge",
                                ),
                            )
                        })?;
                }
                // Inspect and admit the observed slot allocation while it is
                // still empty, before attempting the independently fallible
                // duplicate-check bitmap reserve.
                let bitmap_words = observed.maximum_slots.div_ceil(u64::BITS as usize);
                let mut source_authority_bitmap = Vec::new();
                if resident_injected_allocation_failure() {
                    return Err(GpuBabBoundResidentAdmissionError::Allocation(
                        resident_allocation_error(bitmap_words, "resident source-authority bitmap"),
                    ));
                }
                source_authority_bitmap
                    .try_reserve_exact(bitmap_words)
                    .map_err(|_| {
                        GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                            bitmap_words,
                            "resident source-authority bitmap",
                        ))
                    })?;
                finish_resident_validation(deadline, "resident source-authority bitmap reserve")
                    .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                let bitmap_storage_bytes = source_authority_bitmap
                    .capacity()
                    .checked_mul(size_of::<u64>())
                    .ok_or_else(|| {
                        GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                            bitmap_words,
                            "resident source-authority bitmap capacity charge",
                        ))
                    })?;
                let observed_physical_ledger_bytes = slots
                    .capacity()
                    .checked_mul(size_of::<GpuBabBoundResidentSlotState>())
                    .and_then(|bytes| bytes.checked_add(bitmap_storage_bytes))
                    .ok_or_else(|| {
                        GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                            observed.maximum_slots,
                            "resident configured ledger physical storage",
                        ))
                    })?;
                if observed_physical_ledger_bytes > observed_slot_table_charge {
                    return Err(GpuBabBoundResidentAdmissionError::Allocation(
                        resident_allocation_error(
                            observed.maximum_slots,
                            "resident configured ledger physical storage",
                        ),
                    ));
                }
                // The observed slot-table allocation is within both the
                // immutable policy and the cumulative admission budget while
                // still empty. Only now may core initialize and publish the
                // stable all-Vacant generation-zero table.
                for index in 0..observed.maximum_slots {
                    poll_resident_validation(deadline, index, "resident slot-table initialization")
                        .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                    slots.push(GpuBabBoundResidentSlotState::Vacant { high_generation: 0 });
                }
                finish_resident_validation(deadline, "resident slot-table initialization")
                    .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                for index in 0..bitmap_words {
                    poll_resident_validation(
                        deadline,
                        index,
                        "resident source-authority bitmap initialization",
                    )
                    .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                    source_authority_bitmap.push(0);
                }
                finish_resident_validation(
                    deadline,
                    "resident source-authority bitmap initialization",
                )
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                self.slots = slots;
                self.source_authority_bitmap = source_authority_bitmap;
                self.policy_state = GpuBabBoundResidentPolicyState::Installed(observed);
                Ok(())
            }
            GpuBabBoundResidentPolicyState::Unqueried
            | GpuBabBoundResidentPolicyState::Unsupported => {
                self.poison_all();
                Err(GpuBabBoundResidentAdmissionError::Poison(invalid(
                    "retained-domain policy installation lost its observed policy",
                )))
            }
        }
    }

    fn observe_unsupported(&mut self) -> Result<()> {
        match self.policy_state {
            GpuBabBoundResidentPolicyState::Unqueried => {
                self.policy_state = GpuBabBoundResidentPolicyState::Unsupported;
                Ok(())
            }
            GpuBabBoundResidentPolicyState::Unsupported => Ok(()),
            GpuBabBoundResidentPolicyState::Observed(_)
            | GpuBabBoundResidentPolicyState::Installed(_) => {
                self.poison_all();
                Err(invalid(
                    "installed retained-domain policy disappeared from the backend session",
                ))
            }
        }
    }

    fn installed_policy(&self) -> Option<GpuBabBoundResidentDomainPolicy> {
        match self.policy_state {
            GpuBabBoundResidentPolicyState::Installed(policy) => Some(policy),
            _ => None,
        }
    }

    pub(super) fn policy_was_observed(&self) -> bool {
        self.policy_state != GpuBabBoundResidentPolicyState::Unqueried
    }

    /// Copy-only policy comparison used after resource-capable raw calls and
    /// during cleanup, before certainty and registration state are settled.
    fn policy_reobservation_matches(
        &self,
        observed: Option<GpuBabBoundResidentDomainPolicy>,
    ) -> bool {
        match (self.policy_state, observed) {
            (GpuBabBoundResidentPolicyState::Unqueried, _) => true,
            (GpuBabBoundResidentPolicyState::Unsupported, None) => true,
            (
                GpuBabBoundResidentPolicyState::Observed(expected)
                | GpuBabBoundResidentPolicyState::Installed(expected),
                Some(observed),
            ) if expected == observed && observed.is_valid() => true,
            _ => false,
        }
    }

    fn policy_observation_identity_sha256(&self) -> Result<[u8; 32]> {
        let mut hash = Sha256::new();
        hash.update(b"ny.gpu-bab-bound.resident-policy-observation.v2\0");
        match self.policy_state {
            GpuBabBoundResidentPolicyState::Unsupported => hash.update([0]),
            GpuBabBoundResidentPolicyState::Observed(policy)
            | GpuBabBoundResidentPolicyState::Installed(policy) => {
                hash.update([1]);
                hash.update(resident_policy_identity_sha256(policy));
            }
            GpuBabBoundResidentPolicyState::Unqueried => {
                return Err(invalid(
                    "resident policy observation identity was requested before observation",
                ));
            }
        }
        Ok(hash.finalize().into())
    }

    pub(super) fn ledger_audit_with_deadline(
        &self,
        deadline: Option<ResidentValidationDeadline>,
    ) -> Result<GpuBabBoundResidentLedgerAudit> {
        let mut audit = GpuBabBoundResidentLedgerAudit {
            core_host_charged_bytes: self
                .slots
                .capacity()
                .checked_mul(GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES)
                .ok_or_else(|| invalid("resident configured-slot host reserve overflows usize"))?,
            ..GpuBabBoundResidentLedgerAudit::default()
        };
        for (index, state) in self.slots.iter().enumerate() {
            poll_resident_validation(deadline, index, "resident slot-ledger audit")?;
            match state {
                GpuBabBoundResidentSlotState::Vacant { .. } => {}
                GpuBabBoundResidentSlotState::Reserved { generation, layout } => {
                    if *generation == 0 {
                        return Err(invalid(
                            "resident reserved slot has a zero burned generation",
                        ));
                    }
                    audit.reserved_slots = audit
                        .reserved_slots
                        .checked_add(1)
                        .ok_or_else(|| invalid("resident reserved-slot count overflows usize"))?;
                    audit.resident_device_bytes = audit
                        .resident_device_bytes
                        .checked_add(layout.payload_bytes)
                        .ok_or_else(|| invalid("resident device ledger overflows usize"))?;
                    audit.resident_slots = audit
                        .resident_slots
                        .checked_add(1)
                        .ok_or_else(|| invalid("resident slot count overflows usize"))?;
                }
                GpuBabBoundResidentSlotState::Live(slot) => {
                    if slot.in_flight {
                        audit.in_flight_slots =
                            audit.in_flight_slots.checked_add(1).ok_or_else(|| {
                                invalid("resident in-flight slot count overflows usize")
                            })?;
                    }
                    audit.core_host_charged_bytes = audit
                        .core_host_charged_bytes
                        .checked_add(slot.layout.core_host_charged_bytes)
                        .ok_or_else(|| invalid("resident core-host charge overflows usize"))?;
                    audit.history_words = audit
                        .history_words
                        .checked_add(slot.snapshot.history.len())
                        .ok_or_else(|| invalid("resident history word count overflows usize"))?;
                    match slot.presence {
                        GpuBabBoundResidentPresence::Resident => {
                            audit.resident_device_bytes = audit
                                .resident_device_bytes
                                .checked_add(slot.layout.payload_bytes)
                                .ok_or_else(|| invalid("resident device ledger overflows usize"))?;
                            audit.resident_slots = audit
                                .resident_slots
                                .checked_add(1)
                                .ok_or_else(|| invalid("resident slot count overflows usize"))?;
                        }
                        GpuBabBoundResidentPresence::RefreshOnly => {
                            audit.refresh_only_slots =
                                audit.refresh_only_slots.checked_add(1).ok_or_else(|| {
                                    invalid("refresh-only slot count overflows usize")
                                })?;
                        }
                    }
                }
            }
        }
        finish_resident_validation(deadline, "resident slot-ledger audit")?;
        if audit.in_flight_slots != self.in_flight_slots
            || audit.reserved_slots != self.reserved_slots
        {
            return Err(invalid(
                "resident O(1) transaction counters disagree with the slot ledger",
            ));
        }
        Ok(audit)
    }

    pub(super) fn close_ledger_audit(&self) -> Result<GpuBabBoundResidentLedgerAudit> {
        self.ledger_audit_with_deadline(None)
    }

    #[cfg(test)]
    pub(super) fn resident_bytes(&self) -> Result<usize> {
        Ok(self.ledger_audit_with_deadline(None)?.resident_device_bytes)
    }

    #[cfg(test)]
    pub(super) fn live_counts(&self) -> (usize, usize) {
        let audit = self
            .ledger_audit_with_deadline(None)
            .expect("resident ledger counters remain checked below fixed slot caps");
        (audit.resident_slots, audit.refresh_only_slots)
    }

    #[cfg(test)]
    pub(super) fn core_host_charged_bytes(&self) -> Result<usize> {
        Ok(self
            .ledger_audit_with_deadline(None)?
            .core_host_charged_bytes)
    }

    #[cfg(test)]
    pub(super) fn history_words(&self) -> Result<usize> {
        Ok(self.ledger_audit_with_deadline(None)?.history_words)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuBabBoundResidentConsumedKind {
    Parent,
    Release,
    Evict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuBabBoundResidentConsumedSlot {
    slot_index: usize,
    kind: GpuBabBoundResidentConsumedKind,
    presence: GpuBabBoundResidentPresence,
    resident_bytes: usize,
    core_host_charged_bytes: usize,
    transcript: GpuBabBoundResidentSlotTranscript,
    source_audit: GpuBabBoundResidentSourceAudit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuBabBoundResidentJournalEntry {
    Source { consumed_index: usize },
    Destination { destination_index: usize },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GpuBabBoundResidentAllocationPrefix {
    allocated_bytes: usize,
    complete_slots: usize,
}

#[derive(Debug)]
struct GpuBabBoundPendingResidentWave {
    session_nonce_sha256: [u8; 32],
    policy: GpuBabBoundResidentDomainPolicy,
    prepared_groups: Vec<GpuBabBoundPreparedResidentGroup>,
    prepared_release: Vec<GpuBabBoundResidentSourceAudit>,
    prepared_evict: Vec<GpuBabBoundResidentSourceAudit>,
    destinations: Vec<GpuBabBoundResidentDestinationPlan>,
    destination_snapshots: Vec<GpuBabBoundResidentDomainSnapshot>,
    accepted_destinations: Vec<GpuBabBoundAcceptedResidentDomain>,
    destination_tokens: Vec<GpuBabBoundResidentSlotRef>,
    evicted_tokens: Vec<GpuBabBoundResidentSlotRef>,
    consumed: Vec<GpuBabBoundResidentConsumedSlot>,
    retained_before_bytes: usize,
    destination_bytes: usize,
    released_on_commit_bytes: usize,
    retained_after_bytes: usize,
    host_audit: GpuBabBoundResidentHostAudit,
    resident_slots_before: usize,
    refresh_only_slots_before: usize,
    resident_slots_after: usize,
    refresh_only_slots_after: usize,
    fresh_domains: usize,
    delta_domains: usize,
    consumed_parent_slots: usize,
    explicitly_released_slots: usize,
    explicitly_evicted_slots: usize,
    destination_buffer_units: usize,
    planned_memory: GpuBabBoundResidentMemoryReceipt,
    planned_transfers: GpuBabBoundResidentTransferReceipt,
    transfer_prefixes: Vec<GpuBabBoundResidentTransferReceipt>,
    allocation_prefixes: Vec<GpuBabBoundResidentAllocationPrefix>,
    journal: Vec<GpuBabBoundResidentJournalEntry>,
    in_flight_slots_before: usize,
    reserved_slots_before: usize,
    in_flight_slots_during: usize,
    reserved_slots_during: usize,
    next_completed_waves: u64,
    schedule_identity_sha256: [u8; 32],
}

const _: () = assert!(
    size_of::<GpuBabBoundResidentDomainSnapshot>()
        <= GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES
);
// Every valid wave has at least one destination. One of the five deliberately
// conservative 4-KiB per-destination metadata units therefore covers the
// fixed pending header (including Vec headers and the sealed Copy receipt),
// while the backing capacities are charged by their explicit multiplicities.
const _: () = assert!(
    size_of::<GpuBabBoundPendingResidentWave>() <= GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES
);
const _: () = assert!(
    size_of::<GpuBabBoundResidentDestinationPlan>()
        <= GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES
);
const _: () = assert!(
    size_of::<GpuBabBoundAcceptedResidentDomain>()
        <= GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES
);
const _: () = assert!(
    size_of::<GpuBabBoundResidentSlotRef>() <= GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES
);
const _: () = assert!(
    size_of::<GpuBabBoundPreparedResidentGroup>()
        <= GPU_BAB_BOUND_HOST_PENDING_GROUP_METADATA_BYTES
);
const _: () = assert!(
    size_of::<GpuBabBoundResidentConsumedSlot>()
        <= GPU_BAB_BOUND_HOST_PENDING_SOURCE_METADATA_BYTES
);
const _: () = assert!(
    size_of::<GpuBabBoundResidentSourceAudit>() <= GPU_BAB_BOUND_HOST_PENDING_SOURCE_METADATA_BYTES
);
const _: () =
    assert!(size_of::<SplitLiteralKey>() <= GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES);
const _: () =
    assert!(size_of::<SplitLiteralRecord>() <= GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES);

impl GpuBabBoundPendingResidentWave {
    fn host_audit(&self, completed: bool) -> GpuBabBoundResidentHostAudit {
        if completed {
            self.host_audit
        } else {
            GpuBabBoundResidentHostAudit {
                retained_v2_core_host_after_charged_bytes: self
                    .host_audit
                    .retained_v2_core_host_before_charged_bytes,
                history_after_words: self.host_audit.history_before_words,
                ..self.host_audit
            }
        }
    }

    fn expected_transfers(&self) -> Result<GpuBabBoundResidentTransferReceipt> {
        Ok(self.planned_transfers)
    }

    fn recompute_transfers_with_deadline(
        &self,
        deadline: Option<ResidentValidationDeadline>,
    ) -> Result<GpuBabBoundResidentTransferReceipt> {
        let mut receipt = GpuBabBoundResidentTransferReceipt::default();
        for (index, destination) in self.destinations.iter().enumerate() {
            poll_resident_validation(deadline, index, "resident transfer-plan validation")?;
            let fresh = destination
                .family_transfers
                .iter()
                .all(|source| *source == GpuBabBoundResidentFamilyTransfer::FreshUpload);
            for family in RESIDENT_F32_FAMILIES {
                let bytes = destination.layout.family_payload_bytes[family.index()];
                if bytes != 0 {
                    receipt.resident_transfer_units = receipt
                        .resident_transfer_units
                        .checked_add(1)
                        .ok_or_else(|| invalid("resident transfer-unit count overflows usize"))?;
                    if destination.family_transfers[family.index()]
                        != GpuBabBoundResidentFamilyTransfer::CopyParent
                    {
                        receipt.resident_host_to_device_transfer_units = receipt
                            .resident_host_to_device_transfer_units
                            .checked_add(1)
                            .ok_or_else(|| {
                                invalid("resident H2D transfer-unit count overflows usize")
                            })?;
                    }
                }
                let target = match destination.family_transfers[family.index()] {
                    GpuBabBoundResidentFamilyTransfer::FreshUpload => {
                        &mut receipt.fresh_family_host_to_device_bytes[family.index()]
                    }
                    GpuBabBoundResidentFamilyTransfer::FreshReplace => {
                        &mut receipt.replaced_family_host_to_device_bytes[family.index()]
                    }
                    GpuBabBoundResidentFamilyTransfer::CopyParent => {
                        &mut receipt.copied_family_device_to_device_bytes[family.index()]
                    }
                };
                checked_add_to(target, bytes, "resident family transfer total")?;
            }
            if fresh {
                if destination
                    .history_prefix_bytes
                    .checked_add(destination.history_suffix_bytes)
                    .ok_or_else(|| invalid("fresh history H2D bytes overflow usize"))?
                    != 0
                {
                    receipt.resident_transfer_units = receipt
                        .resident_transfer_units
                        .checked_add(1)
                        .ok_or_else(|| invalid("resident transfer-unit count overflows usize"))?;
                    receipt.resident_host_to_device_transfer_units = receipt
                        .resident_host_to_device_transfer_units
                        .checked_add(1)
                        .ok_or_else(|| {
                            invalid("resident H2D transfer-unit count overflows usize")
                        })?;
                }
                checked_add_to(
                    &mut receipt.fresh_history_host_to_device_bytes,
                    destination
                        .history_prefix_bytes
                        .checked_add(destination.history_suffix_bytes)
                        .ok_or_else(|| invalid("fresh history H2D bytes overflow usize"))?,
                    "fresh history H2D total",
                )?;
            } else {
                receipt.resident_transfer_units = receipt
                    .resident_transfer_units
                    .checked_add(usize::from(destination.history_prefix_bytes != 0))
                    .and_then(|value| {
                        value.checked_add(usize::from(destination.history_suffix_bytes != 0))
                    })
                    .ok_or_else(|| invalid("resident transfer-unit count overflows usize"))?;
                receipt.resident_host_to_device_transfer_units = receipt
                    .resident_host_to_device_transfer_units
                    .checked_add(usize::from(destination.history_suffix_bytes != 0))
                    .ok_or_else(|| invalid("resident H2D transfer-unit count overflows usize"))?;
                checked_add_to(
                    &mut receipt.delta_history_host_to_device_bytes,
                    destination.history_suffix_bytes,
                    "delta history H2D total",
                )?;
                checked_add_to(
                    &mut receipt.history_device_to_device_bytes,
                    destination.history_prefix_bytes,
                    "history D2D total",
                )?;
            }
        }
        finish_resident_validation(deadline, "resident transfer-plan validation")?;
        receipt.resident_host_to_device_bytes = receipt
            .fresh_family_host_to_device_bytes
            .iter()
            .chain(receipt.replaced_family_host_to_device_bytes.iter())
            .copied()
            .try_fold(0usize, |total, bytes| total.checked_add(bytes))
            .and_then(|value| value.checked_add(receipt.fresh_history_host_to_device_bytes))
            .and_then(|value| value.checked_add(receipt.delta_history_host_to_device_bytes))
            .ok_or_else(|| invalid("resident H2D total overflows usize"))?;
        receipt.resident_device_to_device_bytes = receipt
            .copied_family_device_to_device_bytes
            .iter()
            .copied()
            .try_fold(0usize, |total, bytes| total.checked_add(bytes))
            .and_then(|value| value.checked_add(receipt.history_device_to_device_bytes))
            .ok_or_else(|| invalid("resident D2D total overflows usize"))?;
        receipt.completed_resident_transfer_units = receipt.resident_transfer_units;
        Ok(receipt)
    }

    #[cfg(test)]
    fn host_to_device_transfer_units(&self) -> usize {
        self.destinations
            .iter()
            .map(|destination| {
                let family_units = RESIDENT_F32_FAMILIES
                    .iter()
                    .filter(|&&family| {
                        destination.layout.family_payload_bytes[family.index()] != 0
                            && destination.family_transfers[family.index()]
                                != GpuBabBoundResidentFamilyTransfer::CopyParent
                    })
                    .count();
                let fresh = destination
                    .family_transfers
                    .iter()
                    .all(|source| *source == GpuBabBoundResidentFamilyTransfer::FreshUpload);
                let history_h2d_bytes = if fresh {
                    destination
                        .history_prefix_bytes
                        .saturating_add(destination.history_suffix_bytes)
                } else {
                    destination.history_suffix_bytes
                };
                family_units + usize::from(history_h2d_bytes != 0)
            })
            .sum()
    }

    fn transfer_prefix(
        &self,
        completed_units: usize,
    ) -> Result<GpuBabBoundResidentTransferReceipt> {
        self.transfer_prefixes
            .get(completed_units)
            .copied()
            .ok_or_else(|| invalid("resident transfer prefix exceeds the admitted unit plan"))
    }

    fn expected_memory(
        &self,
        completed: bool,
        destination_allocated: bool,
    ) -> GpuBabBoundResidentMemoryReceipt {
        if completed && destination_allocated {
            return self.planned_memory;
        }
        let allocated_destination_bytes = if destination_allocated {
            self.destination_bytes
        } else {
            0
        };
        let allocated_destination_slots = if destination_allocated {
            self.destinations.len()
        } else {
            self.allocation_prefixes[0].complete_slots
        };
        let allocated_destination_buffer_units = if destination_allocated {
            self.destination_buffer_units
        } else {
            0
        };
        Self::memory_receipt(
            self,
            completed,
            self.consumed_parent_slots,
            self.explicitly_released_slots,
            self.explicitly_evicted_slots,
            allocated_destination_bytes,
            allocated_destination_slots,
            allocated_destination_buffer_units,
        )
    }

    #[cfg(test)]
    fn destination_buffer_units(&self) -> usize {
        self.destination_buffer_units
    }

    #[allow(clippy::too_many_arguments)]
    fn memory_receipt(
        &self,
        completed: bool,
        consumed_parent_slots: usize,
        explicitly_released_slots: usize,
        explicitly_evicted_slots: usize,
        allocated_destination_bytes: usize,
        allocated_destination_slots: usize,
        allocated_destination_buffer_units: usize,
    ) -> GpuBabBoundResidentMemoryReceipt {
        GpuBabBoundResidentMemoryReceipt {
            resident_device_before_bytes: self.retained_before_bytes,
            reserved_destination_bytes: self.destination_bytes,
            allocated_destination_bytes,
            released_provisional_destination_bytes: if completed {
                0
            } else {
                allocated_destination_bytes
            },
            planned_release_bytes: self.released_on_commit_bytes,
            committed_release_bytes: if completed {
                self.released_on_commit_bytes
            } else {
                0
            },
            resident_device_after_bytes: if completed {
                self.retained_after_bytes
            } else {
                self.retained_before_bytes
            },
            resident_queued_upload_bytes: 0,
            transition_peak_device_bytes: 0,
            resident_slots_before: self.resident_slots_before,
            refresh_only_slots_before: self.refresh_only_slots_before,
            destination_slots: self.destinations.len(),
            destination_buffer_units: self.destination_buffer_units,
            allocated_destination_slots,
            allocated_destination_buffer_units,
            released_provisional_destination_slots: if completed {
                0
            } else {
                allocated_destination_slots
            },
            released_provisional_destination_buffer_units: if completed {
                0
            } else {
                allocated_destination_buffer_units
            },
            consumed_parent_slots,
            explicitly_released_slots,
            explicitly_evicted_slots,
            resident_slots_after: if completed {
                self.resident_slots_after
            } else {
                self.resident_slots_before
            },
            refresh_only_slots_after: if completed {
                self.refresh_only_slots_after
            } else {
                self.refresh_only_slots_before
            },
            destination_padding_bytes: 0,
        }
    }
}

#[derive(Debug)]
struct GpuBabBoundSealedPendingResidentWave {
    plan: GpuBabBoundPendingResidentWave,
    predispatch_failure_receipt: GpuBabBoundBackendResidentWaveReceipt,
}

impl std::ops::Deref for GpuBabBoundSealedPendingResidentWave {
    type Target = GpuBabBoundPendingResidentWave;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

impl std::ops::DerefMut for GpuBabBoundSealedPendingResidentWave {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.plan
    }
}

impl GpuBabBoundSealedPendingResidentWave {
    fn predispatch_failure_receipt(&self) -> GpuBabBoundBackendResidentWaveReceipt {
        self.predispatch_failure_receipt
    }
}

const _: () = assert!(
    size_of::<GpuBabBoundSealedPendingResidentWave>()
        <= GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES
);

#[derive(Debug)]
struct GpuBabBoundPendingResidentMaintenance {
    session_nonce_sha256: [u8; 32],
    policy: GpuBabBoundResidentDomainPolicy,
    release: Vec<GpuBabBoundResidentSourceAudit>,
    evict: Vec<GpuBabBoundResidentSourceAudit>,
    consumed: Vec<GpuBabBoundResidentConsumedSlot>,
    evicted_tokens: Vec<GpuBabBoundResidentSlotRef>,
    retained_before_bytes: usize,
    released_resident_bytes: usize,
    retained_after_bytes: usize,
    host_audit: GpuBabBoundResidentHostAudit,
    resident_slots_before: usize,
    refresh_only_slots_before: usize,
    resident_slots_after: usize,
    refresh_only_slots_after: usize,
    planned_memory: GpuBabBoundResidentMaintenanceMemoryReceipt,
    in_flight_slots_before: usize,
    reserved_slots_before: usize,
    in_flight_slots_during: usize,
    reserved_slots_during: usize,
    schedule_identity_sha256: [u8; 32],
}

const _: () = {
    assert!(
        size_of::<GpuBabBoundPendingResidentMaintenance>()
            <= GPU_BAB_BOUND_HOST_MAINTENANCE_FIXED_BYTES_PER_OPERATION
    );
    assert!(
        size_of::<GpuBabBoundResidentSlotState>()
            + size_of::<u64>()
            + size_of::<GpuBabBoundResidentConsumedSlot>()
            + size_of::<GpuBabBoundResidentSourceAudit>()
            + size_of::<GpuBabBoundResidentSlotRef>()
            + GPU_BAB_BOUND_HOST_MAINTENANCE_FIXED_BYTES_PER_OPERATION
            <= GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES
    );
};

impl GpuBabBoundPendingResidentMaintenance {
    fn host_audit(&self, completed: bool) -> GpuBabBoundResidentHostAudit {
        if completed {
            self.host_audit
        } else {
            GpuBabBoundResidentHostAudit {
                retained_v2_core_host_after_charged_bytes: self
                    .host_audit
                    .retained_v2_core_host_before_charged_bytes,
                history_after_words: self.host_audit.history_before_words,
                ..self.host_audit
            }
        }
    }

    fn expected_memory(&self, completed: bool) -> GpuBabBoundResidentMaintenanceMemoryReceipt {
        if completed {
            return self.planned_memory;
        }
        GpuBabBoundResidentMaintenanceMemoryReceipt {
            resident_device_before_bytes: self.retained_before_bytes,
            planned_release_device_bytes: self.released_resident_bytes,
            committed_release_device_bytes: if completed {
                self.released_resident_bytes
            } else {
                0
            },
            resident_device_after_bytes: if completed {
                self.retained_after_bytes
            } else {
                self.retained_before_bytes
            },
            resident_slots_before: self.resident_slots_before,
            refresh_only_slots_before: self.refresh_only_slots_before,
            released_slots: self.release.len(),
            evicted_slots: self.evict.len(),
            resident_slots_after: if completed {
                self.resident_slots_after
            } else {
                self.resident_slots_before
            },
            refresh_only_slots_after: if completed {
                self.refresh_only_slots_after
            } else {
                self.refresh_only_slots_before
            },
            destination_slots: 0,
            allocated_destination_slots: 0,
            destination_buffer_units: 0,
            allocated_destination_buffer_units: 0,
            allocated_destination_bytes: 0,
            destination_padding_bytes: 0,
            transition_peak_device_bytes: 0,
        }
    }
}

#[derive(Debug)]
struct GpuBabBoundSealedPendingResidentMaintenance {
    plan: GpuBabBoundPendingResidentMaintenance,
    predispatch_failure_receipt: GpuBabBoundBackendResidentMaintenanceReceipt,
}

impl std::ops::Deref for GpuBabBoundSealedPendingResidentMaintenance {
    type Target = GpuBabBoundPendingResidentMaintenance;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

impl std::ops::DerefMut for GpuBabBoundSealedPendingResidentMaintenance {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.plan
    }
}

impl GpuBabBoundSealedPendingResidentMaintenance {
    fn predispatch_failure_receipt(&self) -> GpuBabBoundBackendResidentMaintenanceReceipt {
        self.predispatch_failure_receipt
    }
}

const _: () = assert!(
    size_of::<GpuBabBoundSealedPendingResidentMaintenance>()
        <= GPU_BAB_BOUND_HOST_MAINTENANCE_FIXED_BYTES_PER_OPERATION
);

#[derive(Debug)]
enum GpuBabBoundResidentAdmissionError {
    Invalid(NyError),
    Decline(GpuBabBoundResidentWaveDecline),
    Allocation(NyError),
    Poison(NyError),
}

fn invalid_admission(message: impl Into<String>) -> GpuBabBoundResidentAdmissionError {
    GpuBabBoundResidentAdmissionError::Invalid(invalid(message))
}

impl std::fmt::Display for GpuBabBoundResidentAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(error) | Self::Allocation(error) | Self::Poison(error) => {
                std::fmt::Display::fmt(error, formatter)
            }
            Self::Decline(reason) => write!(formatter, "resident preaccept decline: {reason:?}"),
        }
    }
}

/// Clean v2 preaccept disposition. Only these variants return the exact owned
/// request and preserve caller fallback/retry authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundResidentWaveDecline {
    Unsupported,
    InsufficientCapacity,
    /// The exact core source is RefreshOnly, so this operation requires a
    /// full upload. This is a mode requirement, not a capacity or lineage-
    /// continuity promise: a full ledger may require validated maintenance
    /// Release followed by a new `FreshUpload { prior: None }` whose complete
    /// provenance is re-proved by the qualified TCB.
    FullRefreshRequired,
    TemporarilyUnavailable,
}

/// Raw backend's allocation-free v2 preflight decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundBackendResidentPrepareDisposition {
    CleanDecline(GpuBabBoundResidentWaveDecline),
    Accepted,
    /// A core-Resident physical source was unexpectedly absent or mismatched.
    /// This is terminal and can never become FullRefreshRequired fallback.
    AuthorityLost,
}

/// Closed raw preflight decision for mandatory zero-destination cleanup.
///
/// Once a session installs a v2 policy, cleanup cannot become unsupported or
/// require capacity/full refresh. Only a bounded retry or acceptance is
/// nonterminal; loss of physical authority is explicit and fail-closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundBackendResidentMaintenancePrepareDisposition {
    TemporarilyUnavailable,
    Accepted,
    AuthorityLost,
}

/// Source class in a backend-facing read-only group transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundResidentSourceClass {
    FreshUpload,
    RetainedDelta,
}

/// Read-only parent group presented to raw preflight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundPreparedResidentGroup {
    parent_group_id: u64,
    prefix: GpuBabBoundArenaRange,
    construction: GpuBabBoundResidentConstruction,
    source_class: GpuBabBoundResidentSourceClass,
    source: Option<GpuBabBoundResidentSourceAudit>,
}

impl GpuBabBoundPreparedResidentGroup {
    #[must_use]
    pub fn parent_group_id(&self) -> u64 {
        self.parent_group_id
    }

    #[must_use]
    pub fn prefix(&self) -> GpuBabBoundArenaRange {
        self.prefix
    }

    #[must_use]
    pub fn construction(&self) -> GpuBabBoundResidentConstruction {
        self.construction
    }

    #[must_use]
    pub fn source_class(&self) -> GpuBabBoundResidentSourceClass {
        self.source_class
    }

    #[must_use]
    pub fn source(&self) -> Option<GpuBabBoundResidentSourceAudit> {
        self.source
    }
}

/// Core-created, read-only v2 raw-preflight context.
pub struct GpuBabBoundPreparedResidentWave<'a> {
    wave: &'a GpuBabBoundWaveRequest,
    split_history: &'a GpuBabBoundSplitHistoryArena,
    domain_histories: &'a [GpuBabBoundSplitHistoryView],
    groups: &'a [GpuBabBoundPreparedResidentGroup],
    release: &'a [GpuBabBoundResidentSourceAudit],
    evict: &'a [GpuBabBoundResidentSourceAudit],
    destinations: &'a [GpuBabBoundProposedResidentDomain],
    planned_memory: GpuBabBoundResidentMemoryReceipt,
    planned_transfers: GpuBabBoundResidentTransferReceipt,
    admission_schedule_identity_sha256: [u8; 32],
    policy: GpuBabBoundResidentDomainPolicy,
}

impl GpuBabBoundPreparedResidentWave<'_> {
    #[must_use]
    pub fn wave(&self) -> &GpuBabBoundWaveRequest {
        self.wave
    }

    #[must_use]
    pub fn split_history(&self) -> &GpuBabBoundSplitHistoryArena {
        self.split_history
    }

    #[must_use]
    /// History view i corresponds exactly to `wave().domains[i]` and
    /// `destinations()[i]` in canonical group-major order.
    pub fn domain_histories(&self) -> &[GpuBabBoundSplitHistoryView] {
        self.domain_histories
    }

    #[must_use]
    pub fn groups(&self) -> &[GpuBabBoundPreparedResidentGroup] {
        self.groups
    }

    #[must_use]
    pub fn release(&self) -> &[GpuBabBoundResidentSourceAudit] {
        self.release
    }

    #[must_use]
    pub fn evict(&self) -> &[GpuBabBoundResidentSourceAudit] {
        self.evict
    }

    #[must_use]
    /// Destination i is exactly `wave().domains[i]` in canonical group-major
    /// domain order. Slot allocation order does not redefine this index.
    pub fn destinations(&self) -> &[GpuBabBoundProposedResidentDomain] {
        self.destinations
    }

    /// Exact completed resident-transition template. Its minimum no-release-
    /// netting peak is open retained graph+phase memory, retained resident
    /// bytes before the wave, full destination liability, and resident H2D
    /// staging. A terminal receipt adds the disjoint accountable v1 wave
    /// working/base-H2D/D2H sum to that minimum.
    #[must_use]
    pub fn planned_memory(&self) -> GpuBabBoundResidentMemoryReceipt {
        self.planned_memory
    }

    #[must_use]
    pub fn planned_transfers(&self) -> GpuBabBoundResidentTransferReceipt {
        self.planned_transfers
    }

    #[must_use]
    pub fn admission_schedule_identity_sha256(&self) -> &[u8; 32] {
        &self.admission_schedule_identity_sha256
    }

    #[must_use]
    pub fn policy(&self) -> GpuBabBoundResidentDomainPolicy {
        self.policy
    }
}

/// Core-created accepted v2 execution context.
pub struct GpuBabBoundAcceptedResidentWave<'a> {
    wave: &'a GpuBabBoundWaveRequest,
    split_history: &'a GpuBabBoundSplitHistoryArena,
    domain_histories: &'a [GpuBabBoundSplitHistoryView],
    groups: &'a [GpuBabBoundPreparedResidentGroup],
    release: &'a [GpuBabBoundResidentSourceAudit],
    evict: &'a [GpuBabBoundResidentSourceAudit],
    destinations: &'a [GpuBabBoundAcceptedResidentDomain],
    planned_memory: GpuBabBoundResidentMemoryReceipt,
    planned_transfers: GpuBabBoundResidentTransferReceipt,
    transcript: GpuBabBoundTerminalTranscript,
    policy: GpuBabBoundResidentDomainPolicy,
}

impl GpuBabBoundAcceptedResidentWave<'_> {
    #[must_use]
    pub fn wave(&self) -> &GpuBabBoundWaveRequest {
        self.wave
    }

    #[must_use]
    pub fn split_history(&self) -> &GpuBabBoundSplitHistoryArena {
        self.split_history
    }

    #[must_use]
    /// Destination i's history corresponds exactly to `wave().domains[i]`
    /// in canonical group-major order.
    pub fn domain_histories(&self) -> &[GpuBabBoundSplitHistoryView] {
        self.domain_histories
    }

    #[must_use]
    pub fn groups(&self) -> &[GpuBabBoundPreparedResidentGroup] {
        self.groups
    }

    #[must_use]
    pub fn release(&self) -> &[GpuBabBoundResidentSourceAudit] {
        self.release
    }

    #[must_use]
    pub fn evict(&self) -> &[GpuBabBoundResidentSourceAudit] {
        self.evict
    }

    #[must_use]
    pub fn planned_memory(&self) -> GpuBabBoundResidentMemoryReceipt {
        self.planned_memory
    }

    #[must_use]
    pub fn planned_transfers(&self) -> GpuBabBoundResidentTransferReceipt {
        self.planned_transfers
    }

    #[must_use]
    /// Accepted destination i is exactly `wave().domains[i]` in canonical
    /// group-major order, independent of its assigned slot index.
    pub fn destinations(&self) -> &[GpuBabBoundAcceptedResidentDomain] {
        self.destinations
    }

    #[must_use]
    pub fn transcript(&self) -> GpuBabBoundTerminalTranscript {
        self.transcript
    }

    #[must_use]
    pub fn policy(&self) -> GpuBabBoundResidentDomainPolicy {
        self.policy
    }
}

/// Core-created, allocation-free maintenance preflight context.
pub struct GpuBabBoundPreparedResidentMaintenance<'a> {
    release: &'a [GpuBabBoundResidentSourceAudit],
    evict: &'a [GpuBabBoundResidentSourceAudit],
    planned_memory: GpuBabBoundResidentMaintenanceMemoryReceipt,
    schedule_identity_sha256: [u8; 32],
    deadline: Instant,
    max_device_bytes: usize,
    policy: GpuBabBoundResidentDomainPolicy,
}

impl GpuBabBoundPreparedResidentMaintenance<'_> {
    #[must_use]
    pub fn release(&self) -> &[GpuBabBoundResidentSourceAudit] {
        self.release
    }

    #[must_use]
    pub fn evict(&self) -> &[GpuBabBoundResidentSourceAudit] {
        self.evict
    }

    #[must_use]
    pub fn planned_memory(&self) -> GpuBabBoundResidentMaintenanceMemoryReceipt {
        self.planned_memory
    }

    #[must_use]
    pub fn schedule_identity_sha256(&self) -> &[u8; 32] {
        &self.schedule_identity_sha256
    }

    #[must_use]
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    #[must_use]
    pub fn max_device_bytes(&self) -> usize {
        self.max_device_bytes
    }

    #[must_use]
    pub fn policy(&self) -> GpuBabBoundResidentDomainPolicy {
        self.policy
    }
}

/// Core-created accepted zero-destination maintenance context.
pub struct GpuBabBoundAcceptedResidentMaintenance<'a> {
    prepared: GpuBabBoundPreparedResidentMaintenance<'a>,
    transcript: GpuBabBoundTerminalTranscript,
}

/// Core-owned conservative host-charge/history audit for one transition.
///
/// These bytes never come from a raw backend receipt: they describe core host
/// ownership. Values are retained-v2 accountable conservative charges, not
/// allocator RSS. They include configured slot reserve, compact v2 snapshots,
/// v2/maintenance pending and publication metadata, and base structural
/// scratch only on a v2 admission path. Generic v1 validation/results,
/// caller/backend/result-owned memory, allocator overhead, and one immediately
/// rejected empty over-cap reserve are excluded. Future Release/Evict is never
/// netted from `peak` at admission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBabBoundResidentHostAudit {
    pub retained_v2_core_host_before_charged_bytes: usize,
    pub retained_v2_core_host_peak_charged_bytes: usize,
    pub retained_v2_core_host_after_charged_bytes: usize,
    pub history_before_words: usize,
    pub history_peak_words: usize,
    pub history_after_words: usize,
}

impl GpuBabBoundAcceptedResidentMaintenance<'_> {
    #[must_use]
    pub fn release(&self) -> &[GpuBabBoundResidentSourceAudit] {
        self.prepared.release
    }

    #[must_use]
    pub fn evict(&self) -> &[GpuBabBoundResidentSourceAudit] {
        self.prepared.evict
    }

    #[must_use]
    pub fn planned_memory(&self) -> GpuBabBoundResidentMaintenanceMemoryReceipt {
        self.prepared.planned_memory
    }

    #[must_use]
    pub fn transcript(&self) -> GpuBabBoundTerminalTranscript {
        self.transcript
    }
}

/// Exact no-compute maintenance residency equation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBabBoundResidentMaintenanceMemoryReceipt {
    pub resident_device_before_bytes: usize,
    pub planned_release_device_bytes: usize,
    pub committed_release_device_bytes: usize,
    pub resident_device_after_bytes: usize,
    pub resident_slots_before: usize,
    pub refresh_only_slots_before: usize,
    pub released_slots: usize,
    pub evicted_slots: usize,
    pub resident_slots_after: usize,
    pub refresh_only_slots_after: usize,
    pub destination_slots: usize,
    pub allocated_destination_slots: usize,
    pub destination_buffer_units: usize,
    pub allocated_destination_buffer_units: usize,
    pub allocated_destination_bytes: usize,
    pub destination_padding_bytes: usize,
    /// Exact peak is open retained residency plus resident bytes before any
    /// release. Future release is never netted from this no-compute peak.
    pub transition_peak_device_bytes: usize,
}

fn hash_resident_maintenance_memory_receipt(
    hash: &mut Sha256,
    receipt: GpuBabBoundResidentMaintenanceMemoryReceipt,
) {
    hash.update(b"resident-maintenance-planned-memory-v1\0");
    for value in [
        receipt.resident_device_before_bytes,
        receipt.planned_release_device_bytes,
        receipt.committed_release_device_bytes,
        receipt.resident_device_after_bytes,
        receipt.resident_slots_before,
        receipt.refresh_only_slots_before,
        receipt.released_slots,
        receipt.evicted_slots,
        receipt.resident_slots_after,
        receipt.refresh_only_slots_after,
        receipt.destination_slots,
        receipt.allocated_destination_slots,
        receipt.destination_buffer_units,
        receipt.allocated_destination_buffer_units,
        receipt.allocated_destination_bytes,
        receipt.destination_padding_bytes,
        receipt.transition_peak_device_bytes,
    ] {
        hash_u64(hash, value as u64);
    }
}

/// Raw maintenance receipt. Every work/transfer field must be exactly zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundBackendResidentMaintenanceReceipt {
    pub transcript: GpuBabBoundTerminalTranscript,
    pub memory: GpuBabBoundResidentMaintenanceMemoryReceipt,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub device_to_device_bytes: usize,
    pub control_payload_bytes: usize,
    pub transfer_units: usize,
    pub completed_transfer_units: usize,
    pub dispatches: usize,
    pub submits: usize,
    pub synchronizations: usize,
    pub readbacks: usize,
}

/// Raw all-terminal maintenance disposition.
pub enum GpuBabBoundBackendResidentMaintenanceDisposition {
    Completed {
        receipt: GpuBabBoundBackendResidentMaintenanceReceipt,
    },
    AcceptedFailure {
        kind: GpuBabBoundBackendFailureKind,
        detail: String,
        receipt: GpuBabBoundBackendResidentMaintenanceReceipt,
    },
    DeadlineExpired {
        detail: String,
        receipt: GpuBabBoundBackendResidentMaintenanceReceipt,
    },
}

/// Maintenance preaccept result; every nonterminal returns the exact request.
#[must_use = "resident maintenance owns non-Clone slot authority"]
pub enum GpuBabBoundResidentMaintenancePreparation<'lease, 'backend> {
    InvalidRequest {
        error: NyError,
        request: GpuBabBoundResidentMaintenanceRequest,
    },
    CleanDecline {
        reason: GpuBabBoundResidentWaveDecline,
        request: GpuBabBoundResidentMaintenanceRequest,
    },
    AcceptedFailure(GpuBabBoundResidentMaintenanceFailure),
    DeadlineExpired(GpuBabBoundResidentMaintenanceFailure),
    Accepted(GpuBabBoundResidentMaintenanceCapability<'lease, 'backend>),
    SessionTerminal(GpuBabBoundSessionTerminal),
}

impl GpuBabBoundResidentMaintenancePreparation<'_, '_> {
    #[must_use]
    pub fn permits_legacy_fallback(&self) -> bool {
        false
    }

    #[must_use]
    pub fn permits_retry(&self) -> bool {
        matches!(self, Self::CleanDecline { .. })
    }
}

/// Postaccept maintenance failure.
pub struct GpuBabBoundResidentMaintenanceFailure {
    kind: GpuBabBoundTerminalFailureKind,
    detail: String,
    receipt: GpuBabBoundBackendResidentMaintenanceReceipt,
    receipt_validated: bool,
    host_audit: Option<GpuBabBoundResidentHostAudit>,
}

impl GpuBabBoundResidentMaintenanceFailure {
    #[must_use]
    pub fn kind(&self) -> GpuBabBoundTerminalFailureKind {
        self.kind
    }

    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }

    #[must_use]
    pub fn receipt(&self) -> &GpuBabBoundBackendResidentMaintenanceReceipt {
        &self.receipt
    }

    #[must_use]
    /// A true value on an AcceptedFailure certifies only exact cleanup-receipt
    /// equality after clean destruction. It never implies restored terminal
    /// authority, registration release, or fallback/retry permission.
    pub fn receipt_validated(&self) -> bool {
        self.receipt_validated
    }

    #[must_use]
    /// Exact core-host ledger audit when the terminal transition settled
    /// without an unwind that could have moved only part of the snapshot
    /// journal. `None` means the core charge/history frontier is quarantined
    /// and must not be treated as authoritative.
    pub fn host_audit(&self) -> Option<GpuBabBoundResidentHostAudit> {
        self.host_audit
    }
}

/// Validated completed maintenance result.
pub struct ValidatedGpuBabBoundResidentMaintenanceResult {
    receipt: GpuBabBoundBackendResidentMaintenanceReceipt,
    evicted_slots: Vec<GpuBabBoundResidentSlotRef>,
    host_audit: GpuBabBoundResidentHostAudit,
}

impl ValidatedGpuBabBoundResidentMaintenanceResult {
    #[must_use]
    pub fn receipt(&self) -> &GpuBabBoundBackendResidentMaintenanceReceipt {
        &self.receipt
    }

    #[must_use]
    /// Tokens preserve the maintenance request's strictly ascending Evict-
    /// section order. Release-section tokens are consumed and not returned.
    pub fn evicted_slots(&self) -> &[GpuBabBoundResidentSlotRef] {
        &self.evicted_slots
    }

    #[must_use]
    pub fn host_audit(&self) -> GpuBabBoundResidentHostAudit {
        self.host_audit
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        GpuBabBoundBackendResidentMaintenanceReceipt,
        GpuBabBoundResidentHostAudit,
        Vec<GpuBabBoundResidentSlotRef>,
    ) {
        (self.receipt, self.host_audit, self.evicted_slots)
    }
}

/// Mandatory postaccept maintenance terminal.
#[must_use = "accepted maintenance has no fallback authority"]
pub enum GpuBabBoundResidentMaintenanceDisposition {
    Completed(ValidatedGpuBabBoundResidentMaintenanceResult),
    AcceptedFailure(GpuBabBoundResidentMaintenanceFailure),
    DeadlineExpired(GpuBabBoundResidentMaintenanceFailure),
}

/// Non-cloneable exact-once maintenance capability.
#[must_use = "dropping accepted maintenance poisons every resident slot"]
pub struct GpuBabBoundResidentMaintenanceCapability<'lease, 'backend> {
    lease: &'lease mut GpuBabBoundPhaseLease<'backend>,
    request: Option<GpuBabBoundResidentMaintenanceRequest>,
    pending: Option<GpuBabBoundSealedPendingResidentMaintenance>,
    transcript: GpuBabBoundTerminalTranscript,
    execution_started: bool,
    executed: bool,
}

/// Exact per-family/history v2 H2D and D2D accounting.
///
/// Array order is Activation, Beta, Abs, BoxLower, BoxUpper, CachedLa.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBabBoundResidentTransferReceipt {
    pub fresh_family_host_to_device_bytes: [usize; 6],
    pub replaced_family_host_to_device_bytes: [usize; 6],
    pub copied_family_device_to_device_bytes: [usize; 6],
    pub fresh_history_host_to_device_bytes: usize,
    pub delta_history_host_to_device_bytes: usize,
    pub history_device_to_device_bytes: usize,
    /// Must remain zero in v2; command-encoder metadata is host-only.
    pub resident_control_payload_bytes: usize,
    /// Total canonical nonzero transfer units in the admitted plan.
    /// The order is every H2D unit destination-major first, followed by every
    /// D2D parent-copy unit destination-major. A predispatch failure can thus
    /// report any fallible H2D prefix without claiming an executed D2D copy.
    pub resident_transfer_units: usize,
    /// Exact phase frontier within `resident_transfer_units`: units before
    /// this index are all H2D, and later units are all D2D. This is core-
    /// derived, schedule-bound, and directly visible to raw providers.
    pub resident_host_to_device_transfer_units: usize,
    /// Completed prefix of the two-phase canonical plan: all destination-major
    /// H2D units first, then all destination-major D2D units.
    pub completed_resident_transfer_units: usize,
    pub resident_host_to_device_bytes: usize,
    pub resident_device_to_device_bytes: usize,
}

/// Exact logical-residency transition receipt for one accepted v2 wave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBabBoundResidentMemoryReceipt {
    pub resident_device_before_bytes: usize,
    pub reserved_destination_bytes: usize,
    pub allocated_destination_bytes: usize,
    pub released_provisional_destination_bytes: usize,
    pub planned_release_bytes: usize,
    pub committed_release_bytes: usize,
    pub resident_device_after_bytes: usize,
    /// Staging allocation for the exact resident H2D payload. It is disjoint
    /// from destination residency in this v2 accounting model.
    pub resident_queued_upload_bytes: usize,
    /// In Prepared/Accepted views this is the minimum no-release-netting
    /// resident peak: open retained graph+phase memory, retained-before, the
    /// certified full destination liability, and resident upload staging. In a
    /// terminal receipt it is the exact physical peak after adding the
    /// disjoint accountable v1 wave working/base-H2D/D2H sum and using the
    /// actually allocated canonical destination-buffer prefix. Future source
    /// releases are never netted from admission or terminal peak accounting.
    pub transition_peak_device_bytes: usize,
    pub resident_slots_before: usize,
    pub refresh_only_slots_before: usize,
    pub destination_slots: usize,
    /// Canonical destination-major prefix units: nonzero Activation, Beta,
    /// Abs, BoxLower, BoxUpper, CachedLa, then History for every slot.
    pub destination_buffer_units: usize,
    pub allocated_destination_slots: usize,
    pub allocated_destination_buffer_units: usize,
    pub released_provisional_destination_slots: usize,
    pub released_provisional_destination_buffer_units: usize,
    pub consumed_parent_slots: usize,
    pub explicitly_released_slots: usize,
    pub explicitly_evicted_slots: usize,
    pub resident_slots_after: usize,
    pub refresh_only_slots_after: usize,
    /// Must remain zero in v2; every destination charge is exact payload.
    pub destination_padding_bytes: usize,
}

fn hash_resident_memory_receipt(hash: &mut Sha256, receipt: GpuBabBoundResidentMemoryReceipt) {
    hash.update(b"resident-planned-memory-v1\0");
    for value in [
        receipt.resident_device_before_bytes,
        receipt.reserved_destination_bytes,
        receipt.allocated_destination_bytes,
        receipt.released_provisional_destination_bytes,
        receipt.planned_release_bytes,
        receipt.committed_release_bytes,
        receipt.resident_device_after_bytes,
        receipt.resident_queued_upload_bytes,
        receipt.transition_peak_device_bytes,
        receipt.resident_slots_before,
        receipt.refresh_only_slots_before,
        receipt.destination_slots,
        receipt.destination_buffer_units,
        receipt.allocated_destination_slots,
        receipt.allocated_destination_buffer_units,
        receipt.released_provisional_destination_slots,
        receipt.released_provisional_destination_buffer_units,
        receipt.consumed_parent_slots,
        receipt.explicitly_released_slots,
        receipt.explicitly_evicted_slots,
        receipt.resident_slots_after,
        receipt.refresh_only_slots_after,
        receipt.destination_padding_bytes,
    ] {
        hash_u64(hash, value as u64);
    }
}

fn hash_resident_host_audit(hash: &mut Sha256, audit: GpuBabBoundResidentHostAudit) {
    hash.update(b"resident-core-host-audit-v1\0");
    for value in [
        audit.retained_v2_core_host_before_charged_bytes,
        audit.retained_v2_core_host_peak_charged_bytes,
        audit.retained_v2_core_host_after_charged_bytes,
        audit.history_before_words,
        audit.history_peak_words,
        audit.history_after_words,
    ] {
        hash_u64(hash, value as u64);
    }
}

/// Raw all-terminal v2 receipt. `wave.transfers` contains only the disjoint
/// nonresident inherited-endpoint/objective/subchunk H2D and result D2H
/// classes; its six domain-family byte fields are zero. `resident_transfers`
/// owns every six-family/history H2D and D2D byte exactly once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundBackendResidentWaveReceipt {
    pub wave: GpuBabBoundBackendWaveReceipt,
    pub resident_memory: GpuBabBoundResidentMemoryReceipt,
    pub resident_transfers: GpuBabBoundResidentTransferReceipt,
    pub fresh_domains: usize,
    pub delta_domains: usize,
}

/// Raw postaccept v2 disposition. A postaccept decline is always illegal.
pub enum GpuBabBoundBackendResidentWaveDisposition {
    Completed {
        domain_outcomes: Vec<GpuBabBoundBackendDomainOutcome>,
        rows: Vec<GpuBabBoundBackendRow>,
        receipt: GpuBabBoundBackendResidentWaveReceipt,
    },
    AcceptedFailure {
        kind: GpuBabBoundBackendFailureKind,
        detail: String,
        receipt: GpuBabBoundBackendResidentWaveReceipt,
    },
    DeadlineExpired {
        detail: String,
        receipt: GpuBabBoundBackendResidentWaveReceipt,
    },
    IllegalCleanDecline {
        reason: GpuBabBoundResidentWaveDecline,
        receipt: GpuBabBoundBackendResidentWaveReceipt,
    },
}

/// V2 preaccept result. Every nonterminal returns the exact owned request.
#[must_use = "resident-wave preparation owns non-Clone source tokens"]
pub enum GpuBabBoundResidentWavePreparation<'lease, 'backend> {
    InvalidRequest {
        error: NyError,
        request: GpuBabBoundResidentWaveRequest,
    },
    CleanDecline {
        reason: GpuBabBoundResidentWaveDecline,
        request: GpuBabBoundResidentWaveRequest,
    },
    AcceptedFailure(GpuBabBoundResidentWaveFailure),
    DeadlineExpired(GpuBabBoundResidentWaveFailure),
    Accepted(GpuBabBoundResidentWaveCapability<'lease, 'backend>),
    SessionTerminal(GpuBabBoundSessionTerminal),
}

impl GpuBabBoundResidentWavePreparation<'_, '_> {
    #[must_use]
    pub fn permits_legacy_fallback(&self) -> bool {
        matches!(self, Self::CleanDecline { .. })
    }
}

/// Core-owned terminal failure for one accepted v2 wave.
pub struct GpuBabBoundResidentWaveFailure {
    kind: GpuBabBoundTerminalFailureKind,
    detail: String,
    receipt: GpuBabBoundBackendResidentWaveReceipt,
    receipt_validated: bool,
    host_audit: Option<GpuBabBoundResidentHostAudit>,
}

impl GpuBabBoundResidentWaveFailure {
    #[must_use]
    pub fn kind(&self) -> GpuBabBoundTerminalFailureKind {
        self.kind
    }

    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }

    #[must_use]
    pub fn receipt(&self) -> &GpuBabBoundBackendResidentWaveReceipt {
        &self.receipt
    }

    #[must_use]
    /// A true value on an AcceptedFailure certifies only exact cleanup-receipt
    /// equality after clean destruction. It never implies restored terminal
    /// authority, registration release, or fallback/retry permission.
    pub fn receipt_validated(&self) -> bool {
        self.receipt_validated
    }

    #[must_use]
    /// Exact core-host ledger audit when the terminal transition settled
    /// without an unwind that could have moved only part of the snapshot
    /// journal. `None` means the core charge/history frontier is quarantined
    /// and must not be treated as authoritative.
    pub fn host_audit(&self) -> Option<GpuBabBoundResidentHostAudit> {
        self.host_audit
    }
}

/// Receipt that passed the v2 transfer/residency validator.
#[derive(Debug, PartialEq, Eq)]
pub struct GpuBabBoundValidatedResidentWaveReceipt {
    raw: GpuBabBoundBackendResidentWaveReceipt,
    host_audit: GpuBabBoundResidentHostAudit,
}

impl GpuBabBoundValidatedResidentWaveReceipt {
    #[must_use]
    pub fn raw_audit_receipt(&self) -> &GpuBabBoundBackendResidentWaveReceipt {
        &self.raw
    }

    #[must_use]
    pub fn core_host_audit(&self) -> GpuBabBoundResidentHostAudit {
        self.host_audit
    }
}

/// Allocation-free owning evidence for core-validated resident outcomes.
/// Outcome i refers to accepted destination i, which is the original
/// `wave.domains[i]` in canonical group-major order.
pub struct GpuBabBoundValidatedResidentDomainOutcomes {
    raw: Vec<GpuBabBoundBackendDomainOutcome>,
}

/// Borrowed validated view of one resident-domain outcome.
#[derive(Clone, Copy)]
pub struct GpuBabBoundValidatedResidentDomainOutcomeRef<'a> {
    raw: &'a GpuBabBoundBackendDomainOutcome,
}

impl<'a> GpuBabBoundValidatedResidentDomainOutcomeRef<'a> {
    #[must_use]
    pub fn parent_group_id(self) -> u64 {
        self.raw.parent_group_id
    }

    #[must_use]
    pub fn child_ordinal(self) -> usize {
        self.raw.child_ordinal
    }

    #[must_use]
    pub fn child_cardinality(self) -> usize {
        self.raw.child_cardinality
    }

    #[must_use]
    pub fn domain_slot(self) -> u64 {
        self.raw.domain_slot
    }

    #[must_use]
    pub fn domain_identity_sha256(self) -> &'a [u8; 32] {
        &self.raw.domain_identity_sha256
    }

    #[must_use]
    pub fn kind(self) -> GpuBabBoundBackendDomainOutcomeKind {
        self.raw.kind
    }
}

impl GpuBabBoundValidatedResidentDomainOutcomes {
    #[must_use]
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<GpuBabBoundValidatedResidentDomainOutcomeRef<'_>> {
        self.raw
            .get(index)
            .map(|raw| GpuBabBoundValidatedResidentDomainOutcomeRef { raw })
    }

    pub fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = GpuBabBoundValidatedResidentDomainOutcomeRef<'_>> {
        self.raw
            .iter()
            .map(|raw| GpuBabBoundValidatedResidentDomainOutcomeRef { raw })
    }

    /// Borrow the in-place storage whose associations were validated by core.
    #[must_use]
    pub fn raw_audit_slice(&self) -> &[GpuBabBoundBackendDomainOutcome] {
        &self.raw
    }

    /// Discard the owning validation evidence and recover raw audit values.
    #[must_use]
    pub fn into_unvalidated_raw(self) -> Vec<GpuBabBoundBackendDomainOutcome> {
        self.raw
    }
}

/// Allocation-free owning evidence for core-validated resident result rows.
/// Rows are destination-major then objective-major: `q = i * R + j` refers to
/// accepted destination i and `wave.objective_indices[j]`.
pub struct GpuBabBoundValidatedResidentRows {
    raw: Vec<GpuBabBoundBackendRow>,
}

/// Borrowed validated view of one resident result row.
#[derive(Clone, Copy)]
pub struct GpuBabBoundValidatedResidentRowRef<'a> {
    raw: &'a GpuBabBoundBackendRow,
}

impl<'a> GpuBabBoundValidatedResidentRowRef<'a> {
    #[must_use]
    pub fn parent_group_id(self) -> u64 {
        self.raw.parent_group_id
    }

    #[must_use]
    pub fn child_ordinal(self) -> usize {
        self.raw.child_ordinal
    }

    #[must_use]
    pub fn child_cardinality(self) -> usize {
        self.raw.child_cardinality
    }

    #[must_use]
    pub fn domain_slot(self) -> u64 {
        self.raw.domain_slot
    }

    #[must_use]
    pub fn domain_identity_sha256(self) -> &'a [u8; 32] {
        &self.raw.domain_identity_sha256
    }

    #[must_use]
    pub fn objective_index(self) -> u32 {
        self.raw.objective_index
    }

    #[must_use]
    pub fn q(self) -> u32 {
        self.raw.q
    }

    #[must_use]
    pub fn lower(self) -> f32 {
        self.raw.lower
    }

    #[must_use]
    pub fn upper(self) -> f32 {
        self.raw.upper
    }
}

impl GpuBabBoundValidatedResidentRows {
    #[must_use]
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<GpuBabBoundValidatedResidentRowRef<'_>> {
        self.raw
            .get(index)
            .map(|raw| GpuBabBoundValidatedResidentRowRef { raw })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = GpuBabBoundValidatedResidentRowRef<'_>> {
        self.raw
            .iter()
            .map(|raw| GpuBabBoundValidatedResidentRowRef { raw })
    }

    /// Borrow the in-place storage whose row/interval/status checks passed.
    #[must_use]
    pub fn raw_audit_slice(&self) -> &[GpuBabBoundBackendRow] {
        &self.raw
    }

    /// Discard the owning validation evidence and recover raw audit values.
    #[must_use]
    pub fn into_unvalidated_raw(self) -> Vec<GpuBabBoundBackendRow> {
        self.raw
    }
}

/// Completed v2 result plus newly issued one-owner resident slot refs.
///
/// ```compile_fail
/// use ny_core::GpuBabBoundResidentSlotRef;
/// let _ = GpuBabBoundResidentSlotRef {
///     session_nonce_sha256: [1; 32],
///     logical_domain_identity_sha256: [2; 32],
///     slot_index: 0,
///     generation: 1,
/// };
/// ```
///
/// ```compile_fail
/// use ny_core::GpuBabBoundResidentSlotRef;
/// fn duplicate(slot: &GpuBabBoundResidentSlotRef) -> GpuBabBoundResidentSlotRef {
///     slot.clone()
/// }
/// ```
pub struct ValidatedGpuBabBoundResidentWaveResult {
    // Raw storage is retained in place to make postaccept validation
    // allocation-free. These values are exposed only through this
    // private-field validated wrapper after allocation-free slice validation.
    domain_outcomes: GpuBabBoundValidatedResidentDomainOutcomes,
    rows: GpuBabBoundValidatedResidentRows,
    receipt: GpuBabBoundValidatedResidentWaveReceipt,
    destination_slots: Vec<GpuBabBoundResidentSlotRef>,
    evicted_slots: Vec<GpuBabBoundResidentSlotRef>,
}

impl ValidatedGpuBabBoundResidentWaveResult {
    #[must_use]
    pub fn domain_outcomes(&self) -> &GpuBabBoundValidatedResidentDomainOutcomes {
        &self.domain_outcomes
    }

    #[must_use]
    pub fn rows(&self) -> &GpuBabBoundValidatedResidentRows {
        &self.rows
    }

    #[must_use]
    pub fn receipt(&self) -> &GpuBabBoundValidatedResidentWaveReceipt {
        &self.receipt
    }

    #[must_use]
    /// Token i owns completed destination i in canonical group-major order.
    pub fn destination_slots(&self) -> &[GpuBabBoundResidentSlotRef] {
        &self.destination_slots
    }

    #[must_use]
    /// Eviction tokens preserve the request's strictly ascending Evict-section
    /// order. Release-section tokens are consumed and are not returned.
    pub fn evicted_slots(&self) -> &[GpuBabBoundResidentSlotRef] {
        &self.evicted_slots
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        GpuBabBoundValidatedResidentDomainOutcomes,
        GpuBabBoundValidatedResidentRows,
        GpuBabBoundValidatedResidentWaveReceipt,
        Vec<GpuBabBoundResidentSlotRef>,
        Vec<GpuBabBoundResidentSlotRef>,
    ) {
        (
            self.domain_outcomes,
            self.rows,
            self.receipt,
            self.destination_slots,
            self.evicted_slots,
        )
    }
}

/// Mandatory terminal after consuming an accepted v2 capability.
#[must_use = "postaccept resident disposition owns slots or terminal failure"]
pub enum GpuBabBoundResidentWaveDisposition {
    Completed(ValidatedGpuBabBoundResidentWaveResult),
    AcceptedFailure(GpuBabBoundResidentWaveFailure),
    DeadlineExpired(GpuBabBoundResidentWaveFailure),
}

impl GpuBabBoundResidentWaveDisposition {
    #[must_use]
    pub fn permits_legacy_fallback(&self) -> bool {
        false
    }
}

/// Non-cloneable exact-once v2 accepted capability.
#[must_use = "dropping an accepted resident capability poisons every slot"]
pub struct GpuBabBoundResidentWaveCapability<'lease, 'backend> {
    lease: &'lease mut GpuBabBoundPhaseLease<'backend>,
    request: Option<GpuBabBoundResidentWaveRequest>,
    shape: ValidatedWaveShape,
    pending: Option<GpuBabBoundSealedPendingResidentWave>,
    transcript: GpuBabBoundTerminalTranscript,
    execution_started: bool,
    executed: bool,
}

impl<'backend> GpuBabBoundPhaseLease<'backend> {
    pub(super) fn recheck_resident_policy_for_close(
        &self,
    ) -> GpuBabBoundResidentClosePolicyRecheck {
        if !self.resident_domains.policy_was_observed() {
            return GpuBabBoundResidentClosePolicyRecheck::Stable;
        }
        let observed = catch_tcb_unwind(|| {
            self.session
                .as_ref()
                .expect("open lease owns a backend session")
                .resident_domain_policy()
        });
        match observed {
            Err(()) => GpuBabBoundResidentClosePolicyRecheck::Panicked,
            Ok(observed) if self.resident_domains.policy_reobservation_matches(observed) => {
                GpuBabBoundResidentClosePolicyRecheck::Stable
            }
            Ok(_) => GpuBabBoundResidentClosePolicyRecheck::Changed,
        }
    }

    /// Observe and latch the session's v2 support policy without allocating
    /// the resident slot table. V1 calls use this before raw preflight so the
    /// support state is stable and transcript-bound even if v2 is never used.
    pub(super) fn observe_resident_policy_for_v1(
        &mut self,
    ) -> std::result::Result<[u8; 32], GpuBabBoundSessionTerminal> {
        let observed = catch_tcb_unwind(|| {
            self.session
                .as_ref()
                .expect("open lease owns a backend session")
                .resident_domain_policy()
        })
        .map_err(|()| GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked)?;
        match observed {
            None => self
                .resident_domains
                .observe_unsupported()
                .map_err(|_| GpuBabBoundSessionTerminal::BackendResidentAuthorityLost)?,
            Some(policy) if !policy.is_valid() => {
                return Err(GpuBabBoundSessionTerminal::InvalidResidentPolicy);
            }
            Some(policy) => self
                .resident_domains
                .observe_supported_policy(policy)
                .map_err(|_| GpuBabBoundSessionTerminal::BackendResidentAuthorityLost)?,
        }
        self.resident_domains
            .policy_observation_identity_sha256()
            .map_err(|_| GpuBabBoundSessionTerminal::BackendResidentAuthorityLost)
    }

    /// Validate and preflight one promotion-grade retained-domain v2 wave.
    /// Every preaccept nonterminal returns the exact non-Clone request.
    pub fn prepare_resident_wave<'lease>(
        &'lease mut self,
        request: GpuBabBoundResidentWaveRequest,
    ) -> GpuBabBoundResidentWavePreparation<'lease, 'backend> {
        if self.state != LeaseState::Open
            || self.resource_certainty != ResidentResourceCertainty::HealthyKnown
        {
            return GpuBabBoundResidentWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::PoisonedOrBusy,
            );
        }
        let validation_deadline =
            ResidentValidationDeadline::new(request.wave.deadline, self.phase.deadline);
        let registration = self.registration;
        let identity = self.transcript.backend;
        let mut entry_guard = match registration.live_guard(identity) {
            Ok(guard) => guard,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        if validation_deadline.expired("resident admission entry deadline gate") {
            entry_guard.poisoned = true;
            self.poison_guarded_registry_with_known_resources();
            drop(entry_guard);
            return GpuBabBoundResidentWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
        }
        drop(entry_guard);

        let policy = catch_tcb_unwind(|| {
            self.session
                .as_ref()
                .expect("open lease owns a backend session")
                .resident_domain_policy()
        });
        let policy = match policy {
            Err(()) => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked,
                );
            }
            Ok(_) if validation_deadline.expired("resident post-policy-query deadline gate") => {
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            Ok(None) => {
                let mut live_guard = match registration.live_guard(identity) {
                    Ok(guard) => guard,
                    Err(_) => {
                        self.resident_domains.poison_all();
                        self.state = LeaseState::Poisoned;
                        self.issuer_claimed = false;
                        return GpuBabBoundResidentWavePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                        );
                    }
                };
                if validation_deadline.expired("resident unsupported deadline gate") {
                    live_guard.poisoned = true;
                    self.poison_guarded_registry_with_known_resources();
                    drop(live_guard);
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                if self.resident_domains.observe_unsupported().is_err() {
                    live_guard.poisoned = true;
                    self.poison_guarded_registry_with_known_resources();
                    drop(live_guard);
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                    );
                }
                let disposition = GpuBabBoundResidentWavePreparation::CleanDecline {
                    reason: GpuBabBoundResidentWaveDecline::Unsupported,
                    request,
                };
                drop(live_guard);
                return disposition;
            }
            Ok(Some(policy)) if policy.is_valid() => {
                if self
                    .resident_domains
                    .observe_supported_policy(policy)
                    .is_err()
                {
                    self.state = LeaseState::Poisoned;
                    self.poison_registry_with_known_resources();
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                    );
                }
                policy
            }
            Ok(Some(_)) => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::InvalidResidentPolicy,
                );
            }
        };

        // Policy observation is intentionally allocation-free and precedes
        // deep request validation. Unsupported therefore wins over a deep
        // InvalidRequest classification. For a supported policy, this checked
        // no-allocation pass charges the prospective configured slot reserve,
        // all live state, compact payload amplification, and every requested
        // validation/plan-container multiplicity before the first workload-
        // scaled core allocation.
        let candidate = match resident_candidate_size(&request, Some(validation_deadline)) {
            Ok(candidate) => candidate,
            Err(NyError::DeadlineExceeded(_)) => {
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            Err(error) => {
                if validation_deadline.expired("resident candidate terminal deadline gate") {
                    self.poison_registry_with_known_resources();
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                return GpuBabBoundResidentWavePreparation::InvalidRequest { error, request };
            }
        };
        let full_refresh_required = match self.resident_domains.prevalidate_source_authority(
            &request,
            self.transcript.backend.session_nonce_sha256,
            validation_deadline,
        ) {
            Ok(required) => required,
            Err(NyError::DeadlineExceeded(_)) => {
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            Err(error) => {
                if validation_deadline.expired("resident source-authority invalid deadline gate") {
                    self.poison_registry_with_known_resources();
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                return GpuBabBoundResidentWavePreparation::InvalidRequest { error, request };
            }
        };
        if full_refresh_required {
            let live_guard = match registration.live_guard(identity) {
                Ok(guard) => guard,
                Err(_) => {
                    self.resident_domains.poison_all();
                    self.state = LeaseState::Poisoned;
                    self.issuer_claimed = false;
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                    );
                }
            };
            if validation_deadline.expired("resident full-refresh deadline gate") {
                let mut live_guard = live_guard;
                live_guard.poisoned = true;
                self.poison_guarded_registry_with_known_resources();
                drop(live_guard);
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            let disposition = GpuBabBoundResidentWavePreparation::CleanDecline {
                reason: GpuBabBoundResidentWaveDecline::FullRefreshRequired,
                request,
            };
            drop(live_guard);
            return disposition;
        }
        let current_ledger = match self
            .resident_domains
            .ledger_audit_with_deadline(Some(validation_deadline))
        {
            Ok(audit) => audit,
            Err(NyError::DeadlineExceeded(_)) => {
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
        };
        let prospective_host_before_bytes = if self.resident_domains.installed_policy().is_some() {
            current_ledger.core_host_charged_bytes
        } else {
            match policy
                .maximum_slots
                .checked_mul(GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES)
            {
                Some(bytes) => bytes,
                None => {
                    self.resident_domains.poison_all();
                    self.state = LeaseState::Poisoned;
                    self.poison_registry();
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::InvalidResidentPolicy,
                    );
                }
            }
        };
        let candidate_host_peak =
            prospective_host_before_bytes.checked_add(candidate.host_transition_charge_bytes);
        let candidate_history_peak = current_ledger
            .history_words
            .checked_add(candidate.history_words);
        let candidate_resident_peak = current_ledger
            .resident_device_bytes
            .checked_add(candidate.logical_payload_bytes);
        if !matches!(candidate_host_peak, Some(bytes) if bytes <= policy.maximum_retained_v2_core_host_charged_bytes)
            || !matches!(candidate_history_peak, Some(words) if words <= policy.maximum_history_words)
            || !matches!(candidate_resident_peak, Some(bytes) if bytes <= policy.maximum_resident_device_bytes)
        {
            let live_guard = match registration.live_guard(identity) {
                Ok(guard) => guard,
                Err(_) => {
                    self.resident_domains.poison_all();
                    self.state = LeaseState::Poisoned;
                    self.issuer_claimed = false;
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                    );
                }
            };
            if validation_deadline.expired("resident capacity-decline deadline gate") {
                let mut live_guard = live_guard;
                live_guard.poisoned = true;
                self.poison_guarded_registry_with_known_resources();
                drop(live_guard);
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            let disposition = GpuBabBoundResidentWavePreparation::CleanDecline {
                reason: GpuBabBoundResidentWaveDecline::InsufficientCapacity,
                request,
            };
            drop(live_guard);
            return disposition;
        }

        let mut host_budget = ResidentHostAdmissionBudget::new(
            policy.maximum_retained_v2_core_host_charged_bytes,
            prospective_host_before_bytes,
            candidate.host_metadata_charge_bytes,
            candidate.logical_payload_bytes,
        )
        .expect("allocation-free resident prepass proved its base host charge is within policy");

        let shape = match request
            .wave
            .validate_for_resident_prepare(&self.phase, &mut host_budget)
        {
            Ok(shape) => shape,
            Err(NyError::DeadlineExceeded(_)) => {
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            Err(error) if error.is_gpu_batch_capacity_exceeded() => {
                let live_guard = match registration.live_guard(identity) {
                    Ok(guard) => guard,
                    Err(_) => {
                        self.resident_domains.poison_all();
                        self.state = LeaseState::Poisoned;
                        self.issuer_claimed = false;
                        return GpuBabBoundResidentWavePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                        );
                    }
                };
                if validation_deadline.expired("resident base-capacity deadline gate") {
                    let mut live_guard = live_guard;
                    live_guard.poisoned = true;
                    self.poison_guarded_registry_with_known_resources();
                    drop(live_guard);
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                let disposition = GpuBabBoundResidentWavePreparation::CleanDecline {
                    reason: GpuBabBoundResidentWaveDecline::InsufficientCapacity,
                    request,
                };
                drop(live_guard);
                return disposition;
            }
            Err(error) => {
                if validation_deadline.expired("resident base-invalid deadline gate") {
                    self.poison_registry_with_known_resources();
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                return GpuBabBoundResidentWavePreparation::InvalidRequest { error, request };
            }
        };
        match self.resident_domains.ensure_policy(
            policy,
            Some(&mut host_budget),
            Some(validation_deadline),
        ) {
            Ok(()) => {}
            Err(GpuBabBoundResidentAdmissionError::Invalid(NyError::DeadlineExceeded(_))) => {
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            Err(GpuBabBoundResidentAdmissionError::Allocation(_)) => {
                let live_guard = match registration.live_guard(identity) {
                    Ok(guard) => guard,
                    Err(_) => {
                        self.resident_domains.poison_all();
                        self.state = LeaseState::Poisoned;
                        self.issuer_claimed = false;
                        return GpuBabBoundResidentWavePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                        );
                    }
                };
                if validation_deadline.expired("resident policy-install capacity deadline gate") {
                    let mut live_guard = live_guard;
                    live_guard.poisoned = true;
                    self.poison_guarded_registry_with_known_resources();
                    drop(live_guard);
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                let disposition = GpuBabBoundResidentWavePreparation::CleanDecline {
                    reason: GpuBabBoundResidentWaveDecline::InsufficientCapacity,
                    request,
                };
                drop(live_guard);
                return disposition;
            }
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::InvalidResidentPolicy,
                );
            }
        }
        let pending = match self.resident_domains.plan_wave(
            &request,
            &self.phase,
            shape,
            candidate,
            self.transcript.backend.session_nonce_sha256,
            self.open_memory,
            &mut host_budget,
            validation_deadline,
        ) {
            Ok(pending) => pending,
            Err(GpuBabBoundResidentAdmissionError::Invalid(NyError::DeadlineExceeded(_))) => {
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            Err(GpuBabBoundResidentAdmissionError::Invalid(error)) => {
                if validation_deadline.expired("resident plan-invalid deadline gate") {
                    self.poison_registry_with_known_resources();
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                return GpuBabBoundResidentWavePreparation::InvalidRequest { error, request };
            }
            Err(GpuBabBoundResidentAdmissionError::Decline(reason)) => {
                let live_guard = match registration.live_guard(identity) {
                    Ok(guard) => guard,
                    Err(_) => {
                        self.resident_domains.poison_all();
                        self.state = LeaseState::Poisoned;
                        self.issuer_claimed = false;
                        return GpuBabBoundResidentWavePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                        );
                    }
                };
                if validation_deadline.expired("resident plan-decline deadline gate") {
                    let mut live_guard = live_guard;
                    live_guard.poisoned = true;
                    self.poison_guarded_registry_with_known_resources();
                    drop(live_guard);
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                let disposition =
                    GpuBabBoundResidentWavePreparation::CleanDecline { reason, request };
                drop(live_guard);
                return disposition;
            }
            Err(GpuBabBoundResidentAdmissionError::Allocation(_)) => {
                let live_guard = match registration.live_guard(identity) {
                    Ok(guard) => guard,
                    Err(_) => {
                        self.resident_domains.poison_all();
                        self.state = LeaseState::Poisoned;
                        self.issuer_claimed = false;
                        return GpuBabBoundResidentWavePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                        );
                    }
                };
                if validation_deadline.expired("resident plan-capacity deadline gate") {
                    let mut live_guard = live_guard;
                    live_guard.poisoned = true;
                    self.poison_guarded_registry_with_known_resources();
                    drop(live_guard);
                    return GpuBabBoundResidentWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                let disposition = GpuBabBoundResidentWavePreparation::CleanDecline {
                    reason: GpuBabBoundResidentWaveDecline::InsufficientCapacity,
                    request,
                };
                drop(live_guard);
                return disposition;
            }
            Err(GpuBabBoundResidentAdmissionError::Poison(_error)) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
        };
        let planned_transfers = match pending.expected_transfers() {
            Ok(transfers) => transfers,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
        };
        let planned_memory = pending.expected_memory(true, true);
        let next_wave_index = match self.last_wave_index.checked_add(1) {
            Some(index) if index != 0 => index,
            _ => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::WaveSequenceExhausted,
                );
            }
        };
        let transcript = GpuBabBoundTerminalTranscript {
            phase: self.transcript,
            wave_index: next_wave_index,
            schedule_identity_sha256: pending.schedule_identity_sha256,
            inherited_endpoints_sha256: shape.inherited_endpoints_sha256,
            deadline: request.wave.deadline,
            max_device_bytes: request.wave.max_device_bytes,
        };
        let predispatch_failure_receipt = match core_predispatch_resident_receipt(
            &request,
            shape,
            &pending,
            transcript,
            self.open_memory,
        ) {
            Ok(receipt) => receipt,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
        };
        let pending = GpuBabBoundSealedPendingResidentWave {
            plan: pending,
            predispatch_failure_receipt,
        };
        let prepared = GpuBabBoundPreparedResidentWave {
            wave: &request.wave,
            split_history: &request.split_history,
            domain_histories: &request.domain_histories,
            groups: &pending.prepared_groups,
            release: &pending.prepared_release,
            evict: &pending.prepared_evict,
            destinations: &pending.accepted_destinations,
            planned_memory,
            planned_transfers,
            admission_schedule_identity_sha256: pending.schedule_identity_sha256,
            policy: pending.policy,
        };
        match self.recheck_resident_policy_for_close() {
            GpuBabBoundResidentClosePolicyRecheck::Stable => {}
            GpuBabBoundResidentClosePolicyRecheck::Changed => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
            GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked,
                );
            }
        }
        let mut pre_raw_guard = match registration.live_guard(identity) {
            Ok(guard) => guard,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        if validation_deadline.expired("resident pre-raw-prepare deadline gate") {
            pre_raw_guard.poisoned = true;
            self.poison_guarded_registry_with_known_resources();
            drop(pre_raw_guard);
            return GpuBabBoundResidentWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
        }
        drop(pre_raw_guard);
        // Raw prepare is contractually pure, but a panicking implementation
        // may have violated that contract. Mark unknown across the call and
        // restore healthy certainty only after a normal return.
        self.mark_resources_unknown();
        let decision = catch_tcb_unwind(|| {
            self.session
                .as_mut()
                .expect("open lease owns a backend session")
                .prepare_resident_wave(&prepared)
        });
        let decision = match decision {
            Ok(decision) => {
                self.mark_resources_healthy_known();
                decision
            }
            Err(()) => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPreparePanicked,
                );
            }
        };
        match self.recheck_resident_policy_for_close() {
            GpuBabBoundResidentClosePolicyRecheck::Stable => {}
            GpuBabBoundResidentClosePolicyRecheck::Changed => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
            GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked,
                );
            }
        }
        let mut live_guard = match registration.live_guard(identity) {
            Ok(guard) => guard,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                return GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        if validation_deadline.expired("resident post-prepare deadline gate") {
            live_guard.poisoned = true;
            self.resident_domains.poison_all();
            self.state = LeaseState::Poisoned;
            self.issuer_claimed = false;
            self.resource_certainty = ResidentResourceCertainty::PoisonedKnown;
            drop(live_guard);
            return GpuBabBoundResidentWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
        }
        match decision {
            GpuBabBoundBackendResidentPrepareDisposition::CleanDecline(
                GpuBabBoundResidentWaveDecline::Unsupported,
            )
            | GpuBabBoundBackendResidentPrepareDisposition::CleanDecline(
                GpuBabBoundResidentWaveDecline::InsufficientCapacity,
            )
            | GpuBabBoundBackendResidentPrepareDisposition::CleanDecline(
                GpuBabBoundResidentWaveDecline::TemporarilyUnavailable,
            ) => {
                let GpuBabBoundBackendResidentPrepareDisposition::CleanDecline(reason) = decision
                else {
                    unreachable!("explicit clean-decline patterns matched")
                };
                let disposition =
                    GpuBabBoundResidentWavePreparation::CleanDecline { reason, request };
                drop(live_guard);
                disposition
            }
            GpuBabBoundBackendResidentPrepareDisposition::Accepted => {
                if validation_deadline.expired("resident pre-reservation deadline gate") {
                    let receipt = pending.predispatch_failure_receipt();
                    live_guard.poisoned = true;
                    self.resident_domains.poison_all();
                    self.state = LeaseState::Poisoned;
                    self.issuer_claimed = false;
                    self.resource_certainty = ResidentResourceCertainty::PoisonedKnown;
                    let disposition = GpuBabBoundResidentWavePreparation::DeadlineExpired(
                        GpuBabBoundResidentWaveFailure {
                            kind: GpuBabBoundTerminalFailureKind::Backend(
                                GpuBabBoundBackendFailureKind::AuthorityLost,
                            ),
                            detail:
                                "resident acceptance expired immediately before slot reservation"
                                    .into(),
                            receipt,
                            receipt_validated: true,
                            host_audit: Some(pending.host_audit(false)),
                        },
                    );
                    drop(live_guard);
                    return disposition;
                }
                let reservation = catch_tcb_unwind(|| {
                    self.resident_domains
                        .reserve_accepted(&pending, Some(validation_deadline))
                });
                if !matches!(&reservation, Ok(Ok(()))) {
                    let receipt = pending.predispatch_failure_receipt();
                    live_guard.poisoned = true;
                    self.resident_domains.poison_all();
                    self.state = LeaseState::Poisoned;
                    self.issuer_claimed = false;
                    let receipt_validated = matches!(&reservation, Ok(Err(_)));
                    let deadline_expired =
                        matches!(&reservation, Ok(Err(NyError::DeadlineExceeded(_))));
                    self.resource_certainty = if receipt_validated {
                        ResidentResourceCertainty::PoisonedKnown
                    } else {
                        ResidentResourceCertainty::PoisonedUnknown
                    };
                    let failure = GpuBabBoundResidentWaveFailure {
                            kind: if deadline_expired {
                                GpuBabBoundTerminalFailureKind::Backend(
                                    GpuBabBoundBackendFailureKind::AuthorityLost,
                                )
                            } else {
                                GpuBabBoundTerminalFailureKind::ContractViolation
                            },
                            detail: if deadline_expired {
                                "resident reservation prevalidation crossed its effective deadline"
                            } else if receipt_validated {
                                "resident reservation rejected an accepted immutable plan before slot mutation"
                            } else {
                                "resident reservation panicked; partial core state is quarantined"
                            }
                            .into(),
                            receipt,
                            receipt_validated,
                            host_audit: Some(pending.host_audit(false)),
                        };
                    let disposition = if deadline_expired {
                        GpuBabBoundResidentWavePreparation::DeadlineExpired(failure)
                    } else {
                        GpuBabBoundResidentWavePreparation::AcceptedFailure(failure)
                    };
                    drop(live_guard);
                    return disposition;
                }
                self.last_wave_index = next_wave_index;
                self.state = LeaseState::WaveAccepted(next_wave_index);
                if validation_deadline.expired("resident post-reservation deadline gate") {
                    let receipt = pending.predispatch_failure_receipt();
                    let rollback_clean =
                        catch_tcb_unwind(|| self.resident_domains.rollback_accepted(&pending))
                            .is_ok();
                    self.resident_domains.poison_all();
                    live_guard.poisoned = true;
                    self.state = LeaseState::Poisoned;
                    self.issuer_claimed = false;
                    self.resource_certainty = if rollback_clean {
                        ResidentResourceCertainty::PoisonedKnown
                    } else {
                        ResidentResourceCertainty::PoisonedUnknown
                    };
                    let disposition = GpuBabBoundResidentWavePreparation::DeadlineExpired(
                        GpuBabBoundResidentWaveFailure {
                            kind: GpuBabBoundTerminalFailureKind::Backend(
                                GpuBabBoundBackendFailureKind::AuthorityLost,
                            ),
                            detail: "resident acceptance expired after slot reservation; the sealed journal was rolled back".into(),
                            receipt,
                            receipt_validated: rollback_clean,
                            host_audit: Some(pending.host_audit(false)),
                        },
                    );
                    drop(live_guard);
                    return disposition;
                }
                let disposition = GpuBabBoundResidentWavePreparation::Accepted(
                    GpuBabBoundResidentWaveCapability {
                        lease: self,
                        request: Some(request),
                        shape,
                        pending: Some(pending),
                        transcript,
                        execution_started: false,
                        executed: false,
                    },
                );
                drop(live_guard);
                disposition
            }
            GpuBabBoundBackendResidentPrepareDisposition::AuthorityLost
            | GpuBabBoundBackendResidentPrepareDisposition::CleanDecline(
                GpuBabBoundResidentWaveDecline::FullRefreshRequired,
            ) => {
                live_guard.poisoned = true;
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                drop(live_guard);
                GpuBabBoundResidentWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                )
            }
        }
    }

    /// Validate and preflight one zero-destination release/eviction transaction.
    pub fn prepare_resident_maintenance<'lease>(
        &'lease mut self,
        request: GpuBabBoundResidentMaintenanceRequest,
    ) -> GpuBabBoundResidentMaintenancePreparation<'lease, 'backend> {
        if self.state != LeaseState::Open
            || self.resource_certainty != ResidentResourceCertainty::HealthyKnown
        {
            return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::PoisonedOrBusy,
            );
        }
        let validation_deadline =
            ResidentValidationDeadline::new(request.deadline, self.phase.deadline);
        let registration = self.registration;
        let identity = self.transcript.backend;
        let mut entry_guard = match registration.live_guard(identity) {
            Ok(guard) => guard,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        if validation_deadline.expired("maintenance admission entry deadline gate") {
            entry_guard.poisoned = true;
            self.poison_guarded_registry_with_known_resources();
            drop(entry_guard);
            return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
        }
        if request.deadline > self.phase.deadline {
            drop(entry_guard);
            return GpuBabBoundResidentMaintenancePreparation::InvalidRequest {
                error: invalid("resident maintenance deadline exceeds its phase deadline"),
                request,
            };
        }
        drop(entry_guard);
        let policy = catch_tcb_unwind(|| {
            self.session
                .as_ref()
                .expect("open lease owns a backend session")
                .resident_domain_policy()
        });
        let policy = match policy {
            Err(()) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked,
                );
            }
            Ok(None) => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
            Ok(Some(policy)) if policy.is_valid() => {
                if self
                    .resident_domains
                    .observe_supported_policy(policy)
                    .is_err()
                {
                    self.resident_domains.poison_all();
                    self.state = LeaseState::Poisoned;
                    self.poison_registry_with_known_resources();
                    return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::InvalidResidentPolicy,
                    );
                }
                match self.resident_domains.prevalidate_maintenance_authority(
                    &request,
                    policy.maximum_slots,
                    identity.session_nonce_sha256,
                    validation_deadline,
                ) {
                    Ok(_) => {}
                    Err(NyError::DeadlineExceeded(_)) => {
                        self.poison_registry_with_known_resources();
                        return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                        );
                    }
                    Err(error) => {
                        if validation_deadline
                            .expired("maintenance authority-invalid deadline gate")
                        {
                            self.poison_registry_with_known_resources();
                            return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                            );
                        }
                        return GpuBabBoundResidentMaintenancePreparation::InvalidRequest {
                            error,
                            request,
                        };
                    }
                }
                match self
                    .resident_domains
                    .ensure_policy(policy, None, Some(validation_deadline))
                {
                    Ok(()) => {}
                    Err(GpuBabBoundResidentAdmissionError::Invalid(NyError::DeadlineExceeded(
                        _,
                    ))) => {
                        self.poison_registry_with_known_resources();
                        return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                        );
                    }
                    Err(GpuBabBoundResidentAdmissionError::Allocation(_)) => {
                        let live_guard = match registration.live_guard(identity) {
                            Ok(guard) => guard,
                            Err(_) => {
                                self.resident_domains.poison_all();
                                self.state = LeaseState::Poisoned;
                                self.issuer_claimed = false;
                                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                                );
                            }
                        };
                        if validation_deadline
                            .expired("maintenance policy-install capacity deadline gate")
                        {
                            let mut live_guard = live_guard;
                            live_guard.poisoned = true;
                            self.poison_guarded_registry_with_known_resources();
                            drop(live_guard);
                            return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                            );
                        }
                        let disposition = GpuBabBoundResidentMaintenancePreparation::CleanDecline {
                            reason: GpuBabBoundResidentWaveDecline::TemporarilyUnavailable,
                            request,
                        };
                        drop(live_guard);
                        return disposition;
                    }
                    Err(_) => {
                        self.resident_domains.poison_all();
                        self.state = LeaseState::Poisoned;
                        self.poison_registry_with_known_resources();
                        return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::InvalidResidentPolicy,
                        );
                    }
                }
                policy
            }
            Ok(Some(_)) => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::InvalidResidentPolicy,
                );
            }
        };
        if validation_deadline.expired("maintenance post-policy deadline gate") {
            self.poison_registry_with_known_resources();
            return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
        }
        let pending = match self.resident_domains.plan_maintenance(
            &request,
            &self.phase,
            identity.session_nonce_sha256,
            self.open_memory,
            validation_deadline,
        ) {
            Ok(pending) => pending,
            Err(GpuBabBoundResidentAdmissionError::Invalid(NyError::DeadlineExceeded(_))) => {
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            Err(GpuBabBoundResidentAdmissionError::Invalid(error)) => {
                if validation_deadline.expired("maintenance plan-invalid deadline gate") {
                    self.poison_registry_with_known_resources();
                    return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                return GpuBabBoundResidentMaintenancePreparation::InvalidRequest {
                    error,
                    request,
                };
            }
            Err(GpuBabBoundResidentAdmissionError::Decline(_))
            | Err(GpuBabBoundResidentAdmissionError::Poison(_)) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
            Err(GpuBabBoundResidentAdmissionError::Allocation(_)) => {
                let live_guard = match registration.live_guard(identity) {
                    Ok(guard) => guard,
                    Err(_) => {
                        self.resident_domains.poison_all();
                        self.state = LeaseState::Poisoned;
                        self.issuer_claimed = false;
                        return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                        );
                    }
                };
                if validation_deadline.expired("maintenance plan-capacity deadline gate") {
                    let mut live_guard = live_guard;
                    live_guard.poisoned = true;
                    self.poison_guarded_registry_with_known_resources();
                    drop(live_guard);
                    return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                let disposition = GpuBabBoundResidentMaintenancePreparation::CleanDecline {
                    reason: GpuBabBoundResidentWaveDecline::TemporarilyUnavailable,
                    request,
                };
                drop(live_guard);
                return disposition;
            }
        };
        let peak = self.open_memory.checked_sum().and_then(|value| {
            value
                .checked_add(pending.retained_before_bytes)
                .ok_or_else(|| invalid("resident maintenance peak overflows usize"))
        });
        let peak = match peak {
            Ok(value) if value <= self.phase.max_device_bytes => value,
            _ => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
        };
        let next_wave_index = match self.last_wave_index.checked_add(1) {
            Some(index) if index != 0 => index,
            _ => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::WaveSequenceExhausted,
                );
            }
        };
        let transcript = GpuBabBoundTerminalTranscript {
            phase: self.transcript,
            wave_index: next_wave_index,
            schedule_identity_sha256: pending.schedule_identity_sha256,
            inherited_endpoints_sha256: maintenance_endpoints_identity(),
            deadline: request.deadline,
            max_device_bytes: self.phase.max_device_bytes,
        };
        let predispatch_failure_receipt = match core_resident_maintenance_receipt(
            &pending,
            transcript,
            self.open_memory,
            false,
        ) {
            Ok(receipt) => receipt,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
        };
        let pending = GpuBabBoundSealedPendingResidentMaintenance {
            plan: pending,
            predispatch_failure_receipt,
        };
        let mut planned_memory = pending.expected_memory(true);
        planned_memory.transition_peak_device_bytes = peak;
        let prepared = GpuBabBoundPreparedResidentMaintenance {
            release: &pending.release,
            evict: &pending.evict,
            planned_memory,
            schedule_identity_sha256: pending.schedule_identity_sha256,
            deadline: request.deadline,
            max_device_bytes: self.phase.max_device_bytes,
            policy,
        };
        match self.recheck_resident_policy_for_close() {
            GpuBabBoundResidentClosePolicyRecheck::Stable => {}
            GpuBabBoundResidentClosePolicyRecheck::Changed => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
            GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked,
                );
            }
        }
        let mut pre_raw_guard = match registration.live_guard(identity) {
            Ok(guard) => guard,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        if validation_deadline.expired("maintenance pre-raw-prepare deadline gate") {
            pre_raw_guard.poisoned = true;
            self.poison_guarded_registry_with_known_resources();
            drop(pre_raw_guard);
            return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
        }
        drop(pre_raw_guard);
        self.mark_resources_unknown();
        let decision = catch_tcb_unwind(|| {
            self.session
                .as_mut()
                .expect("open lease owns a backend session")
                .prepare_resident_maintenance(&prepared)
        });
        let decision = match decision {
            Ok(decision) => {
                self.mark_resources_healthy_known();
                decision
            }
            Err(()) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPreparePanicked,
                );
            }
        };
        match self.recheck_resident_policy_for_close() {
            GpuBabBoundResidentClosePolicyRecheck::Stable => {}
            GpuBabBoundResidentClosePolicyRecheck::Changed => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
            GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked,
                );
            }
        }
        let mut live_guard = match registration.live_guard(identity) {
            Ok(guard) => guard,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        if validation_deadline.expired("maintenance post-prepare deadline gate") {
            live_guard.poisoned = true;
            self.resident_domains.poison_all();
            self.state = LeaseState::Poisoned;
            self.issuer_claimed = false;
            self.resource_certainty = ResidentResourceCertainty::PoisonedKnown;
            drop(live_guard);
            return GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
        }
        match decision {
            GpuBabBoundBackendResidentMaintenancePrepareDisposition::TemporarilyUnavailable => {
                let disposition = GpuBabBoundResidentMaintenancePreparation::CleanDecline {
                    reason: GpuBabBoundResidentWaveDecline::TemporarilyUnavailable,
                    request,
                };
                drop(live_guard);
                disposition
            }
            GpuBabBoundBackendResidentMaintenancePrepareDisposition::Accepted => {
                if validation_deadline.expired("maintenance pre-reservation deadline gate") {
                    let receipt = pending.predispatch_failure_receipt();
                    live_guard.poisoned = true;
                    self.resident_domains.poison_all();
                    self.state = LeaseState::Poisoned;
                    self.issuer_claimed = false;
                    self.resource_certainty = ResidentResourceCertainty::PoisonedKnown;
                    let disposition =
                        GpuBabBoundResidentMaintenancePreparation::DeadlineExpired(
                            GpuBabBoundResidentMaintenanceFailure {
                                kind: GpuBabBoundTerminalFailureKind::Backend(
                                    GpuBabBoundBackendFailureKind::AuthorityLost,
                                ),
                                detail: "maintenance acceptance expired immediately before source reservation".into(),
                                receipt,
                                receipt_validated: true,
                                host_audit: Some(pending.host_audit(false)),
                            },
                        );
                    drop(live_guard);
                    return disposition;
                }
                let reservation = catch_tcb_unwind(|| {
                    self.resident_domains
                        .reserve_maintenance(&pending, Some(validation_deadline))
                });
                if !matches!(&reservation, Ok(Ok(()))) {
                    let receipt = pending.predispatch_failure_receipt();
                    live_guard.poisoned = true;
                    self.resident_domains.poison_all();
                    self.state = LeaseState::Poisoned;
                    self.issuer_claimed = false;
                    let receipt_validated = matches!(&reservation, Ok(Err(_)));
                    let deadline_expired =
                        matches!(&reservation, Ok(Err(NyError::DeadlineExceeded(_))));
                    self.resource_certainty = if receipt_validated {
                        ResidentResourceCertainty::PoisonedKnown
                    } else {
                        ResidentResourceCertainty::PoisonedUnknown
                    };
                    let failure = GpuBabBoundResidentMaintenanceFailure {
                                kind: if deadline_expired {
                                    GpuBabBoundTerminalFailureKind::Backend(
                                        GpuBabBoundBackendFailureKind::AuthorityLost,
                                    )
                                } else {
                                    GpuBabBoundTerminalFailureKind::ContractViolation
                                },
                                detail: if deadline_expired {
                                    "maintenance reservation prevalidation crossed its effective deadline"
                                } else if receipt_validated {
                                    "maintenance reservation rejected an accepted immutable plan before slot mutation"
                                } else {
                                    "maintenance reservation panicked; partial core state is quarantined"
                                }
                                .into(),
                                receipt,
                                receipt_validated,
                                host_audit: Some(pending.host_audit(false)),
                            };
                    let disposition = if deadline_expired {
                        GpuBabBoundResidentMaintenancePreparation::DeadlineExpired(failure)
                    } else {
                        GpuBabBoundResidentMaintenancePreparation::AcceptedFailure(failure)
                    };
                    drop(live_guard);
                    return disposition;
                }
                self.last_wave_index = next_wave_index;
                self.state = LeaseState::WaveAccepted(next_wave_index);
                if validation_deadline.expired("maintenance post-reservation deadline gate") {
                    let receipt = pending.predispatch_failure_receipt();
                    let rollback_clean =
                        catch_tcb_unwind(|| self.resident_domains.rollback_maintenance(&pending))
                            .is_ok();
                    self.resident_domains.poison_all();
                    live_guard.poisoned = true;
                    self.state = LeaseState::Poisoned;
                    self.issuer_claimed = false;
                    self.resource_certainty = if rollback_clean {
                        ResidentResourceCertainty::PoisonedKnown
                    } else {
                        ResidentResourceCertainty::PoisonedUnknown
                    };
                    let disposition =
                        GpuBabBoundResidentMaintenancePreparation::DeadlineExpired(
                            GpuBabBoundResidentMaintenanceFailure {
                                kind: GpuBabBoundTerminalFailureKind::Backend(
                                    GpuBabBoundBackendFailureKind::AuthorityLost,
                                ),
                                detail: "maintenance acceptance expired after source reservation; the sealed journal was rolled back".into(),
                                receipt,
                                receipt_validated: rollback_clean,
                                host_audit: Some(pending.host_audit(false)),
                            },
                        );
                    drop(live_guard);
                    return disposition;
                }
                let disposition = GpuBabBoundResidentMaintenancePreparation::Accepted(
                    GpuBabBoundResidentMaintenanceCapability {
                        lease: self,
                        request: Some(request),
                        pending: Some(pending),
                        transcript,
                        execution_started: false,
                        executed: false,
                    },
                );
                drop(live_guard);
                disposition
            }
            GpuBabBoundBackendResidentMaintenancePrepareDisposition::AuthorityLost => {
                live_guard.poisoned = true;
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                drop(live_guard);
                GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                )
            }
        }
    }

    /// Core-owned terminal retained when a v2 capability was abandoned.
    #[must_use]
    pub fn abandoned_resident_terminal(&self) -> Option<&GpuBabBoundResidentWaveFailure> {
        self.abandoned_resident_terminal.as_ref()
    }

    #[must_use]
    pub fn abandoned_resident_maintenance_terminal(
        &self,
    ) -> Option<&GpuBabBoundResidentMaintenanceFailure> {
        self.abandoned_resident_maintenance_terminal.as_ref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GpuBabBoundResidentClosePolicyRecheck {
    Stable,
    Changed,
    Panicked,
}

impl GpuBabBoundResidentMaintenanceCapability<'_, '_> {
    /// Execute exactly one accepted zero-destination cleanup transaction.
    pub fn execute_accepted(mut self) -> GpuBabBoundResidentMaintenanceDisposition {
        let request = self
            .request
            .as_ref()
            .expect("unexecuted maintenance capability owns its request");
        let pending = self
            .pending
            .as_ref()
            .expect("unexecuted maintenance capability owns its plan");
        let validation_deadline =
            ResidentValidationDeadline::new(request.deadline, self.lease.phase.deadline);
        let registration = self.lease.registration;
        let identity = self.transcript.phase.backend;

        let mut live_guard = match registration.live_guard_noalloc(identity) {
            Some(guard) => guard,
            None => {
                let disposition = rollback_maintenance_before_raw_failure(
                    &mut *self.lease,
                    pending,
                    self.transcript,
                    "maintenance registration authority was lost before raw execution",
                    false,
                );
                self.executed = true;
                return disposition;
            }
        };
        if validation_deadline.expired("maintenance capability entry deadline gate") {
            let receipt = pending.predispatch_failure_receipt();
            let host_audit = pending.host_audit(false);
            let rollback_known =
                guarded_rollback_maintenance(&mut self.lease.resident_domains, pending);
            self.lease.resident_domains.poison_all();
            live_guard.poisoned = true;
            self.lease.state = LeaseState::Poisoned;
            self.lease.issuer_claimed = false;
            self.lease.resource_certainty = if rollback_known {
                ResidentResourceCertainty::PoisonedKnown
            } else {
                ResidentResourceCertainty::PoisonedUnknown
            };
            self.executed = true;
            drop(live_guard);
            return if rollback_known {
                GpuBabBoundResidentMaintenanceDisposition::DeadlineExpired(
                    GpuBabBoundResidentMaintenanceFailure {
                        kind: GpuBabBoundTerminalFailureKind::Backend(
                            GpuBabBoundBackendFailureKind::AuthorityLost,
                        ),
                        detail: "accepted maintenance capability expired before raw execution"
                            .into(),
                        receipt,
                        receipt_validated: true,
                        host_audit: Some(host_audit),
                    },
                )
            } else {
                maintenance_contract_failure(
                    "maintenance rollback panicked before raw execution".into(),
                    receipt,
                    host_audit,
                )
            };
        }
        drop(live_guard);

        let planned_memory = pending.planned_memory;
        let accepted = GpuBabBoundAcceptedResidentMaintenance {
            prepared: GpuBabBoundPreparedResidentMaintenance {
                release: &pending.release,
                evict: &pending.evict,
                planned_memory,
                schedule_identity_sha256: pending.schedule_identity_sha256,
                deadline: request.deadline,
                max_device_bytes: self.lease.phase.max_device_bytes,
                policy: pending.policy,
            },
            transcript: self.transcript,
        };

        match self.lease.recheck_resident_policy_for_close() {
            GpuBabBoundResidentClosePolicyRecheck::Stable => {}
            GpuBabBoundResidentClosePolicyRecheck::Changed => {
                let disposition = rollback_maintenance_before_raw_failure(
                    &mut *self.lease,
                    pending,
                    self.transcript,
                    "maintenance policy changed before raw execution",
                    true,
                );
                self.executed = true;
                return disposition;
            }
            GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                let disposition = rollback_maintenance_before_raw_failure(
                    &mut *self.lease,
                    pending,
                    self.transcript,
                    "maintenance policy recheck panicked before raw execution",
                    false,
                );
                self.executed = true;
                return disposition;
            }
        }

        let mut final_guard = match registration.live_guard_noalloc(identity) {
            Some(guard) => guard,
            None => {
                let disposition = rollback_maintenance_before_raw_failure(
                    &mut *self.lease,
                    pending,
                    self.transcript,
                    "maintenance authority was lost before raw execution",
                    false,
                );
                self.executed = true;
                return disposition;
            }
        };
        if validation_deadline.expired("maintenance capability pre-execute deadline gate") {
            let receipt = pending.predispatch_failure_receipt();
            let host_audit = pending.host_audit(false);
            let rollback_known =
                guarded_rollback_maintenance(&mut self.lease.resident_domains, pending);
            self.lease.resident_domains.poison_all();
            final_guard.poisoned = true;
            self.lease.state = LeaseState::Poisoned;
            self.lease.issuer_claimed = false;
            self.lease.resource_certainty = if rollback_known {
                ResidentResourceCertainty::PoisonedKnown
            } else {
                ResidentResourceCertainty::PoisonedUnknown
            };
            self.executed = true;
            drop(final_guard);
            return if rollback_known {
                GpuBabBoundResidentMaintenanceDisposition::DeadlineExpired(
                    GpuBabBoundResidentMaintenanceFailure {
                        kind: GpuBabBoundTerminalFailureKind::Backend(
                            GpuBabBoundBackendFailureKind::AuthorityLost,
                        ),
                        detail: "maintenance expired at the final raw-execution gate".into(),
                        receipt,
                        receipt_validated: true,
                        host_audit: Some(host_audit),
                    },
                )
            } else {
                maintenance_contract_failure(
                    "maintenance rollback panicked at the final raw-execution gate".into(),
                    receipt,
                    host_audit,
                )
            };
        }
        drop(final_guard);

        self.lease.mark_resources_unknown();
        self.execution_started = true;
        let raw = catch_tcb_unwind(|| {
            self.lease
                .session
                .as_mut()
                .expect("accepted maintenance owns a raw session")
                .execute_accepted_resident_maintenance(&accepted)
        });
        let raw = match raw {
            Ok(raw) => raw,
            Err(()) => {
                self.lease.resident_domains.poison_all();
                self.lease.state = LeaseState::Poisoned;
                self.lease.poison_registry();
                let receipt = pending.predispatch_failure_receipt();
                self.executed = true;
                return maintenance_contract_failure(
                    "raw maintenance execution panicked; physical release state is unknown".into(),
                    receipt,
                    pending.host_audit(false),
                );
            }
        };
        let (Some(request), Some(mut pending)) = (self.request.take(), self.pending.take()) else {
            let receipt = raw_maintenance_receipt(&raw);
            self.lease.resident_domains.poison_all();
            self.lease.state = LeaseState::Poisoned;
            self.lease.poison_registry();
            self.executed = true;
            return maintenance_contract_failure_without_host_audit(
                "executed maintenance capability lost its sealed request or plan".into(),
                receipt,
            );
        };
        let disposition = finish_accepted_resident_maintenance(
            &mut *self.lease,
            &request,
            &mut pending,
            self.transcript,
            raw,
        );
        self.executed = true;
        disposition
    }
}

impl Drop for GpuBabBoundResidentMaintenanceCapability<'_, '_> {
    fn drop(&mut self) {
        if self.executed {
            return;
        }
        let registration = self.lease.registration;
        let identity = self.transcript.phase.backend;
        let mut live_guard = registration.live_guard_noalloc(identity);
        let evidence = self.pending.as_ref().map(|pending| {
            (
                pending.predispatch_failure_receipt(),
                pending.host_audit(false),
            )
        });
        let rollback_clean = !self.execution_started
            && self.pending.as_ref().is_some_and(|pending| {
                guarded_rollback_maintenance(&mut self.lease.resident_domains, pending)
            });
        let resources_known = rollback_clean && live_guard.is_some();
        self.lease.resident_domains.poison_all();
        if let Some(guard) = live_guard.as_mut() {
            guard.poisoned = true;
        }
        self.lease.state = LeaseState::Poisoned;
        self.lease.issuer_claimed = false;
        self.lease.resource_certainty = if resources_known {
            ResidentResourceCertainty::PoisonedKnown
        } else {
            ResidentResourceCertainty::PoisonedUnknown
        };
        self.lease.abandoned_resident_maintenance_terminal = evidence.map(
            |(receipt, host_audit)| GpuBabBoundResidentMaintenanceFailure {
                kind: if resources_known {
                    GpuBabBoundTerminalFailureKind::CapabilityAbandoned
                } else {
                    GpuBabBoundTerminalFailureKind::ContractViolation
                },
                detail: if self.execution_started {
                    "accepted maintenance unwound without an exact physical release receipt".into()
                } else {
                    "accepted maintenance capability was abandoned before execution".into()
                },
                receipt,
                receipt_validated: resources_known,
                host_audit: Some(host_audit),
            },
        );
    }
}

impl GpuBabBoundResidentWaveCapability<'_, '_> {
    /// Consume and execute exactly one accepted retained-domain transaction.
    /// Every disposition is terminal except a validated `Completed`; fallback
    /// authority can never cross this boundary.
    pub fn execute_accepted(mut self) -> GpuBabBoundResidentWaveDisposition {
        let request = self
            .request
            .as_ref()
            .expect("unexecuted resident capability owns its request");
        let pending = self
            .pending
            .as_ref()
            .expect("unexecuted resident capability owns its pending plan");
        let validation_deadline =
            ResidentValidationDeadline::new(request.wave.deadline, self.lease.phase.deadline);
        let registration = self.lease.registration;
        let identity = self.transcript.phase.backend;
        let mut live_guard = match registration.live_guard_noalloc(identity) {
            Some(guard) => guard,
            None => {
                let disposition = rollback_resident_before_raw_failure(
                    &mut *self.lease,
                    request,
                    self.shape,
                    pending,
                    self.transcript,
                    "live registration authority was lost before resident execution",
                    false,
                );
                self.executed = true;
                return disposition;
            }
        };
        if validation_deadline.expired("resident capability entry deadline gate") {
            let receipt = pending.predispatch_failure_receipt();
            let host_audit = pending.host_audit(false);
            let rollback_known =
                guarded_rollback_resident(&mut self.lease.resident_domains, pending);
            self.lease.resident_domains.poison_all();
            live_guard.poisoned = true;
            self.lease.state = LeaseState::Poisoned;
            self.lease.issuer_claimed = false;
            self.lease.resource_certainty = if rollback_known {
                ResidentResourceCertainty::PoisonedKnown
            } else {
                ResidentResourceCertainty::PoisonedUnknown
            };
            self.executed = true;
            drop(live_guard);
            return if rollback_known {
                GpuBabBoundResidentWaveDisposition::DeadlineExpired(
                    GpuBabBoundResidentWaveFailure {
                        kind: GpuBabBoundTerminalFailureKind::Backend(
                            GpuBabBoundBackendFailureKind::AuthorityLost,
                        ),
                        detail: "accepted resident capability expired before raw execution began"
                            .into(),
                        receipt,
                        receipt_validated: true,
                        host_audit: Some(host_audit),
                    },
                )
            } else {
                resident_contract_failure(
                    "resident rollback panicked before raw execution".into(),
                    receipt,
                    host_audit,
                )
            };
        }
        drop(live_guard);

        let accepted = GpuBabBoundAcceptedResidentWave {
            wave: &request.wave,
            split_history: &request.split_history,
            domain_histories: &request.domain_histories,
            groups: &pending.prepared_groups,
            release: &pending.prepared_release,
            evict: &pending.prepared_evict,
            destinations: &pending.accepted_destinations,
            planned_memory: pending.planned_memory,
            planned_transfers: pending.planned_transfers,
            transcript: self.transcript,
            policy: pending.policy,
        };
        match self.lease.recheck_resident_policy_for_close() {
            GpuBabBoundResidentClosePolicyRecheck::Stable => {}
            GpuBabBoundResidentClosePolicyRecheck::Changed => {
                let disposition = rollback_resident_before_raw_failure(
                    &mut *self.lease,
                    request,
                    self.shape,
                    pending,
                    self.transcript,
                    "resident policy changed before raw execution",
                    true,
                );
                self.executed = true;
                return disposition;
            }
            GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                let disposition = rollback_resident_before_raw_failure(
                    &mut *self.lease,
                    request,
                    self.shape,
                    pending,
                    self.transcript,
                    "resident policy recheck panicked before raw execution",
                    false,
                );
                self.executed = true;
                return disposition;
            }
        }
        // Destination-plan construction may allocate and may be preempted.
        // Reacquire exact authority and recheck both hard deadlines immediately
        // before the first raw call; no post-expiry upload/copy/dispatch starts.
        let mut pre_execute_guard = match registration.live_guard_noalloc(identity) {
            Some(guard) => guard,
            None => {
                let disposition = rollback_resident_before_raw_failure(
                    &mut *self.lease,
                    request,
                    self.shape,
                    pending,
                    self.transcript,
                    "live registration authority was lost immediately before resident execution",
                    false,
                );
                self.executed = true;
                return disposition;
            }
        };
        if validation_deadline.expired("resident capability pre-execute deadline gate") {
            let receipt = pending.predispatch_failure_receipt();
            let host_audit = pending.host_audit(false);
            let rollback_known =
                guarded_rollback_resident(&mut self.lease.resident_domains, pending);
            self.lease.resident_domains.poison_all();
            pre_execute_guard.poisoned = true;
            self.lease.state = LeaseState::Poisoned;
            self.lease.issuer_claimed = false;
            self.lease.resource_certainty = if rollback_known {
                ResidentResourceCertainty::PoisonedKnown
            } else {
                ResidentResourceCertainty::PoisonedUnknown
            };
            self.executed = true;
            drop(pre_execute_guard);
            return if rollback_known {
                GpuBabBoundResidentWaveDisposition::DeadlineExpired(
                    GpuBabBoundResidentWaveFailure {
                        kind: GpuBabBoundTerminalFailureKind::Backend(
                            GpuBabBoundBackendFailureKind::AuthorityLost,
                        ),
                        detail: "resident capability expired at the final raw-execution gate"
                            .into(),
                        receipt,
                        receipt_validated: true,
                        host_audit: Some(host_audit),
                    },
                )
            } else {
                resident_contract_failure(
                    "resident rollback panicked at the final raw-execution gate".into(),
                    receipt,
                    host_audit,
                )
            };
        }
        drop(pre_execute_guard);
        self.lease.mark_resources_unknown();
        self.execution_started = true;
        let raw = catch_tcb_unwind(|| {
            self.lease
                .session
                .as_mut()
                .expect("accepted resident lease owns a raw session")
                .execute_accepted_resident(&accepted)
        });
        let raw = match raw {
            Ok(raw) => raw,
            Err(()) => {
                self.lease.resident_domains.poison_all();
                self.lease.state = LeaseState::Poisoned;
                self.lease.poison_registry();
                let receipt = pending.predispatch_failure_receipt();
                self.executed = true;
                return resident_contract_failure(
                    "raw backend panicked after accepted resident execution; resource state is unknown"
                        .into(),
                    receipt,
                    pending.host_audit(false),
                );
            }
        };
        let (Some(request), Some(mut pending)) = (self.request.take(), self.pending.take()) else {
            let receipt = raw_resident_receipt(&raw);
            self.lease.resident_domains.poison_all();
            self.lease.state = LeaseState::Poisoned;
            self.lease.poison_registry();
            self.executed = true;
            return resident_contract_failure_without_host_audit(
                "executed resident capability lost its sealed request or plan".into(),
                receipt,
            );
        };
        let disposition = finish_accepted_resident_wave(
            &mut *self.lease,
            &request,
            self.shape,
            &mut pending,
            self.transcript,
            raw,
        );
        self.executed = true;
        disposition
    }
}

impl Drop for GpuBabBoundResidentWaveCapability<'_, '_> {
    fn drop(&mut self) {
        if self.executed {
            return;
        }
        let registration = self.lease.registration;
        let identity = self.transcript.phase.backend;
        let mut live_guard = registration.live_guard_noalloc(identity);
        let evidence = self.pending.as_ref().map(|pending| {
            (
                pending.predispatch_failure_receipt(),
                pending.host_audit(false),
            )
        });
        let rollback_clean = !self.execution_started
            && self.pending.as_ref().is_some_and(|pending| {
                guarded_rollback_resident(&mut self.lease.resident_domains, pending)
            });
        let resources_known = rollback_clean && live_guard.is_some();
        self.lease.resident_domains.poison_all();
        if let Some(guard) = live_guard.as_mut() {
            guard.poisoned = true;
        }
        self.lease.state = LeaseState::Poisoned;
        self.lease.issuer_claimed = false;
        self.lease.resource_certainty = if resources_known {
            ResidentResourceCertainty::PoisonedKnown
        } else {
            ResidentResourceCertainty::PoisonedUnknown
        };
        self.lease.abandoned_resident_terminal =
            evidence.map(|(receipt, host_audit)| GpuBabBoundResidentWaveFailure {
                kind: if resources_known {
                    GpuBabBoundTerminalFailureKind::CapabilityAbandoned
                } else {
                    GpuBabBoundTerminalFailureKind::ContractViolation
                },
                detail: if self.execution_started {
                    "accepted resident execution unwound without a trustworthy resource receipt"
                        .into()
                } else {
                    "accepted resident capability dropped before execution".into()
                },
                receipt,
                receipt_validated: resources_known,
                host_audit: Some(host_audit),
            });
    }
}

impl From<NyError> for GpuBabBoundResidentAdmissionError {
    fn from(error: NyError) -> Self {
        Self::Invalid(error)
    }
}

fn checked_add_to(total: &mut usize, amount: usize, label: &str) -> Result<()> {
    *total = total
        .checked_add(amount)
        .ok_or_else(|| invalid(format!("{label} overflows usize")))?;
    Ok(())
}

fn expected_resident_transfers_without_copy(
    request: &GpuBabBoundResidentWaveRequest,
    group_sources: &[Option<(usize, &GpuBabBoundResidentLiveSlot)>],
    deadline: ResidentValidationDeadline,
) -> Result<GpuBabBoundResidentTransferReceipt> {
    if group_sources.len() != request.parent_groups.len() {
        return Err(invalid(
            "resident transfer prepass group-source count is inconsistent",
        ));
    }
    let mut receipt = GpuBabBoundResidentTransferReceipt::default();
    for (group_index, (base_group, resident_group)) in request
        .wave
        .parent_groups
        .iter()
        .zip(request.parent_groups.iter())
        .enumerate()
    {
        deadline.poll(group_index, "resident transfer-plan groups")?;
        let source = group_sources[group_index].map(|(_, live)| live);
        let end_domain = base_group
            .first_domain
            .checked_add(base_group.child_cardinality)
            .ok_or_else(|| invalid("resident transfer-prepass coverage overflows usize"))?;
        let prefix_bytes = resident_group
            .prefix
            .len
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| invalid("resident transfer-prepass prefix bytes overflow usize"))?;
        for domain_index in base_group.first_domain..end_domain {
            deadline.poll(domain_index, "resident transfer-plan domains")?;
            let domain = &request.wave.domains[domain_index];
            let family_values = [
                domain.operands.activation.slice(
                    request.wave.domain_arena.activation.as_ref(),
                    "resident activation",
                )?,
                domain
                    .operands
                    .beta
                    .slice(request.wave.domain_arena.beta.as_ref(), "resident beta")?,
                domain
                    .operands
                    .abs
                    .slice(request.wave.domain_arena.abs.as_ref(), "resident abs")?,
                domain.operands.box_lower.slice(
                    request.wave.domain_arena.box_lower.as_ref(),
                    "resident box lower",
                )?,
                domain.operands.box_upper.slice(
                    request.wave.domain_arena.box_upper.as_ref(),
                    "resident box upper",
                )?,
                domain.operands.cached_la.slice(
                    request.wave.domain_arena.cached_la.as_ref(),
                    "resident cached-lA",
                )?,
            ];
            for family in RESIDENT_F32_FAMILIES {
                let values = family_values[family.index()];
                let bytes = values
                    .len()
                    .checked_mul(size_of::<f32>())
                    .ok_or_else(|| invalid("resident family prepass bytes overflow usize"))?;
                let (target, host_to_device) = if resident_group.source.is_delta() {
                    let live = source.ok_or_else(|| {
                        invalid("retained delta transfer prepass has no live source")
                    })?;
                    if f32_bits_equal_with_deadline(values, live.snapshot.family(family), deadline)?
                    {
                        (
                            &mut receipt.copied_family_device_to_device_bytes[family.index()],
                            false,
                        )
                    } else {
                        (
                            &mut receipt.replaced_family_host_to_device_bytes[family.index()],
                            true,
                        )
                    }
                } else {
                    (
                        &mut receipt.fresh_family_host_to_device_bytes[family.index()],
                        true,
                    )
                };
                if bytes != 0 {
                    receipt.resident_transfer_units = receipt
                        .resident_transfer_units
                        .checked_add(1)
                        .ok_or_else(|| invalid("resident transfer-prepass units overflow usize"))?;
                    if host_to_device {
                        receipt.resident_host_to_device_transfer_units = receipt
                            .resident_host_to_device_transfer_units
                            .checked_add(1)
                            .ok_or_else(|| {
                                invalid("resident H2D transfer-prepass units overflow usize")
                            })?;
                    }
                }
                checked_add_to(target, bytes, "resident family transfer prepass")?;
            }
            let suffix_bytes = request.domain_histories[domain_index]
                .suffix
                .len
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| invalid("resident transfer-prepass suffix bytes overflow usize"))?;
            if resident_group.source.is_delta() {
                receipt.resident_transfer_units = receipt
                    .resident_transfer_units
                    .checked_add(usize::from(prefix_bytes != 0))
                    .and_then(|value| value.checked_add(usize::from(suffix_bytes != 0)))
                    .ok_or_else(|| invalid("resident transfer-prepass units overflow usize"))?;
                receipt.resident_host_to_device_transfer_units = receipt
                    .resident_host_to_device_transfer_units
                    .checked_add(usize::from(suffix_bytes != 0))
                    .ok_or_else(|| invalid("resident H2D transfer-prepass units overflow usize"))?;
                checked_add_to(
                    &mut receipt.history_device_to_device_bytes,
                    prefix_bytes,
                    "resident history D2D prepass",
                )?;
                checked_add_to(
                    &mut receipt.delta_history_host_to_device_bytes,
                    suffix_bytes,
                    "resident delta-history H2D prepass",
                )?;
            } else {
                let full_bytes = prefix_bytes
                    .checked_add(suffix_bytes)
                    .ok_or_else(|| invalid("resident fresh-history bytes overflow usize"))?;
                if full_bytes != 0 {
                    receipt.resident_transfer_units = receipt
                        .resident_transfer_units
                        .checked_add(1)
                        .ok_or_else(|| invalid("resident transfer-prepass units overflow usize"))?;
                    receipt.resident_host_to_device_transfer_units = receipt
                        .resident_host_to_device_transfer_units
                        .checked_add(1)
                        .ok_or_else(|| {
                            invalid("resident H2D transfer-prepass units overflow usize")
                        })?;
                }
                checked_add_to(
                    &mut receipt.fresh_history_host_to_device_bytes,
                    full_bytes,
                    "resident fresh-history H2D prepass",
                )?;
            }
        }
    }
    deadline.check("resident transfer plan")?;
    receipt.resident_host_to_device_bytes = receipt
        .fresh_family_host_to_device_bytes
        .iter()
        .chain(receipt.replaced_family_host_to_device_bytes.iter())
        .copied()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .and_then(|value| value.checked_add(receipt.fresh_history_host_to_device_bytes))
        .and_then(|value| value.checked_add(receipt.delta_history_host_to_device_bytes))
        .ok_or_else(|| invalid("resident H2D prepass total overflows usize"))?;
    receipt.resident_device_to_device_bytes = receipt
        .copied_family_device_to_device_bytes
        .iter()
        .copied()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .and_then(|value| value.checked_add(receipt.history_device_to_device_bytes))
        .ok_or_else(|| invalid("resident D2D prepass total overflows usize"))?;
    receipt.completed_resident_transfer_units = receipt.resident_transfer_units;
    Ok(receipt)
}

fn push_resident_transfer_prefix(
    prefixes: &mut Vec<GpuBabBoundResidentTransferReceipt>,
    receipt: &mut GpuBabBoundResidentTransferReceipt,
    bytes: usize,
    host_to_device: bool,
    deadline: ResidentValidationDeadline,
) -> Result<()> {
    receipt.completed_resident_transfer_units = receipt
        .completed_resident_transfer_units
        .checked_add(1)
        .ok_or_else(|| invalid("resident transfer-prefix unit count overflows usize"))?;
    if host_to_device {
        receipt.resident_host_to_device_bytes = receipt
            .resident_host_to_device_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("resident transfer-prefix H2D bytes overflow usize"))?;
    } else {
        receipt.resident_device_to_device_bytes = receipt
            .resident_device_to_device_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("resident transfer-prefix D2D bytes overflow usize"))?;
    }
    deadline.poll(
        receipt.completed_resident_transfer_units,
        "resident transfer-prefix certificate",
    )?;
    prefixes.push(*receipt);
    Ok(())
}

fn build_resident_transfer_prefixes(
    destinations: &[GpuBabBoundResidentDestinationPlan],
    planned: GpuBabBoundResidentTransferReceipt,
    mut prefixes: Vec<GpuBabBoundResidentTransferReceipt>,
    deadline: ResidentValidationDeadline,
) -> Result<Vec<GpuBabBoundResidentTransferReceipt>> {
    if prefixes.capacity() < planned.resident_transfer_units.saturating_add(1) {
        return Err(invalid(
            "resident transfer-prefix certificate capacity is smaller than its sealed plan",
        ));
    }
    let mut receipt = GpuBabBoundResidentTransferReceipt {
        resident_transfer_units: planned.resident_transfer_units,
        resident_host_to_device_transfer_units: planned.resident_host_to_device_transfer_units,
        ..GpuBabBoundResidentTransferReceipt::default()
    };
    prefixes.push(receipt);

    // Canonical phase 1: every fallible H2D unit, destination-major.
    for (destination_index, destination) in destinations.iter().enumerate() {
        deadline.poll(
            destination_index,
            "resident H2D transfer-prefix destinations",
        )?;
        for family in RESIDENT_F32_FAMILIES {
            let bytes = destination.layout.family_payload_bytes[family.index()];
            let target = match destination.family_transfers[family.index()] {
                GpuBabBoundResidentFamilyTransfer::FreshUpload if bytes != 0 => {
                    Some(&mut receipt.fresh_family_host_to_device_bytes[family.index()])
                }
                GpuBabBoundResidentFamilyTransfer::FreshReplace if bytes != 0 => {
                    Some(&mut receipt.replaced_family_host_to_device_bytes[family.index()])
                }
                _ => None,
            };
            if let Some(target) = target {
                *target = target
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("resident H2D family prefix overflows usize"))?;
                push_resident_transfer_prefix(&mut prefixes, &mut receipt, bytes, true, deadline)?;
            }
        }
        let fresh = destination
            .family_transfers
            .iter()
            .all(|source| *source == GpuBabBoundResidentFamilyTransfer::FreshUpload);
        let history_bytes = if fresh {
            destination
                .history_prefix_bytes
                .checked_add(destination.history_suffix_bytes)
                .ok_or_else(|| invalid("resident fresh-history prefix bytes overflow usize"))?
        } else {
            destination.history_suffix_bytes
        };
        if history_bytes != 0 {
            let target = if fresh {
                &mut receipt.fresh_history_host_to_device_bytes
            } else {
                &mut receipt.delta_history_host_to_device_bytes
            };
            *target = target
                .checked_add(history_bytes)
                .ok_or_else(|| invalid("resident H2D history prefix overflows usize"))?;
            push_resident_transfer_prefix(
                &mut prefixes,
                &mut receipt,
                history_bytes,
                true,
                deadline,
            )?;
        }
    }
    deadline.check("resident H2D transfer-prefix certificate")?;
    if receipt.completed_resident_transfer_units != planned.resident_host_to_device_transfer_units {
        return Err(invalid(
            "resident H2D transfer-prefix frontier differs from its sealed plan",
        ));
    }

    // Canonical phase 2: every immutable D2D parent copy, destination-major.
    for (destination_index, destination) in destinations.iter().enumerate() {
        deadline.poll(
            destination_index,
            "resident D2D transfer-prefix destinations",
        )?;
        for family in RESIDENT_F32_FAMILIES {
            let bytes = destination.layout.family_payload_bytes[family.index()];
            if bytes == 0
                || destination.family_transfers[family.index()]
                    != GpuBabBoundResidentFamilyTransfer::CopyParent
            {
                continue;
            }
            receipt.copied_family_device_to_device_bytes[family.index()] = receipt
                .copied_family_device_to_device_bytes[family.index()]
            .checked_add(bytes)
            .ok_or_else(|| invalid("resident D2D family prefix overflows usize"))?;
            push_resident_transfer_prefix(&mut prefixes, &mut receipt, bytes, false, deadline)?;
        }
        let fresh = destination
            .family_transfers
            .iter()
            .all(|source| *source == GpuBabBoundResidentFamilyTransfer::FreshUpload);
        if !fresh && destination.history_prefix_bytes != 0 {
            receipt.history_device_to_device_bytes = receipt
                .history_device_to_device_bytes
                .checked_add(destination.history_prefix_bytes)
                .ok_or_else(|| invalid("resident D2D history prefix overflows usize"))?;
            push_resident_transfer_prefix(
                &mut prefixes,
                &mut receipt,
                destination.history_prefix_bytes,
                false,
                deadline,
            )?;
        }
    }
    deadline.check("resident D2D transfer-prefix certificate")?;
    if receipt != planned || prefixes.len() != planned.resident_transfer_units + 1 {
        return Err(invalid(
            "resident transfer-prefix certificate does not end at the sealed full plan",
        ));
    }
    Ok(prefixes)
}

fn build_resident_allocation_prefixes(
    destinations: &[GpuBabBoundResidentDestinationPlan],
    expected_units: usize,
    mut prefixes: Vec<GpuBabBoundResidentAllocationPrefix>,
    deadline: ResidentValidationDeadline,
) -> Result<Vec<GpuBabBoundResidentAllocationPrefix>> {
    if prefixes.capacity() < expected_units.saturating_add(1) {
        return Err(invalid(
            "resident allocation-prefix certificate capacity is smaller than its sealed plan",
        ));
    }
    prefixes.push(GpuBabBoundResidentAllocationPrefix::default());
    let mut bytes = 0usize;
    let mut complete_slots = 0usize;
    for (destination_index, destination) in destinations.iter().enumerate() {
        deadline.poll(destination_index, "resident allocation-prefix destinations")?;
        let units = destination
            .layout
            .family_payload_bytes
            .into_iter()
            .chain([destination.layout.history_payload_bytes])
            .filter(|&unit_bytes| unit_bytes != 0);
        let unit_count = units.clone().count();
        if unit_count == 0 {
            complete_slots = complete_slots
                .checked_add(1)
                .ok_or_else(|| invalid("resident allocation complete slots overflow usize"))?;
            prefixes
                .last_mut()
                .expect("resident allocation prefix owns its zero entry")
                .complete_slots = complete_slots;
            continue;
        }
        for (unit_index, unit_bytes) in units.enumerate() {
            bytes = bytes
                .checked_add(unit_bytes)
                .ok_or_else(|| invalid("resident allocation-prefix bytes overflow usize"))?;
            if unit_index + 1 == unit_count {
                complete_slots = complete_slots
                    .checked_add(1)
                    .ok_or_else(|| invalid("resident allocation complete slots overflow usize"))?;
            }
            deadline.poll(prefixes.len(), "resident allocation-prefix units")?;
            prefixes.push(GpuBabBoundResidentAllocationPrefix {
                allocated_bytes: bytes,
                complete_slots,
            });
        }
    }
    deadline.check("resident allocation-prefix certificate")?;
    if prefixes.len() != expected_units + 1 {
        return Err(invalid(
            "resident allocation-prefix certificate unit count differs from its sealed plan",
        ));
    }
    Ok(prefixes)
}

fn hash_resident_transfer_receipt(hash: &mut Sha256, receipt: GpuBabBoundResidentTransferReceipt) {
    for bytes in receipt
        .fresh_family_host_to_device_bytes
        .into_iter()
        .chain(receipt.replaced_family_host_to_device_bytes)
        .chain(receipt.copied_family_device_to_device_bytes)
        .chain([
            receipt.fresh_history_host_to_device_bytes,
            receipt.delta_history_host_to_device_bytes,
            receipt.history_device_to_device_bytes,
            receipt.resident_control_payload_bytes,
            receipt.resident_transfer_units,
            receipt.resident_host_to_device_transfer_units,
            receipt.completed_resident_transfer_units,
            receipt.resident_host_to_device_bytes,
            receipt.resident_device_to_device_bytes,
        ])
    {
        hash_u64(hash, bytes as u64);
    }
}

fn resident_terminal_certificate_identity(
    transfer_prefixes: &[GpuBabBoundResidentTransferReceipt],
    allocation_prefixes: &[GpuBabBoundResidentAllocationPrefix],
    deadline: ResidentValidationDeadline,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.resident-terminal-certificate.v1\0");
    hash_u64(&mut hash, transfer_prefixes.len() as u64);
    for (index, receipt) in transfer_prefixes.iter().copied().enumerate() {
        deadline.poll(index, "resident transfer-prefix certificate hash")?;
        hash_resident_transfer_receipt(&mut hash, receipt);
    }
    deadline.check("resident transfer-prefix certificate hash")?;
    hash_u64(&mut hash, allocation_prefixes.len() as u64);
    for (index, prefix) in allocation_prefixes.iter().copied().enumerate() {
        deadline.poll(index, "resident allocation-prefix certificate hash")?;
        hash_u64(&mut hash, prefix.allocated_bytes as u64);
        hash_u64(&mut hash, prefix.complete_slots as u64);
    }
    deadline.check("resident allocation-prefix certificate hash")?;
    Ok(hash.finalize().into())
}

fn resident_terminal_schedule_identity(
    history_schedule_identity_sha256: [u8; 32],
    session_nonce_sha256: [u8; 32],
    destinations: &[GpuBabBoundResidentDestinationPlan],
    consumed: &[GpuBabBoundResidentConsumedSlot],
    planned_transfers: GpuBabBoundResidentTransferReceipt,
    planned_memory: GpuBabBoundResidentMemoryReceipt,
    host_audit: GpuBabBoundResidentHostAudit,
    terminal_certificate_identity_sha256: [u8; 32],
    policy: GpuBabBoundResidentDomainPolicy,
    deadline: ResidentValidationDeadline,
) -> Result<[u8; 32]> {
    let mut schedule_hash = Sha256::new();
    schedule_hash.update(b"ny.gpu-bab-bound.resident-terminal-schedule.v2\0");
    schedule_hash.update(b"transfer-order:all-h2d-then-all-d2d\0");
    schedule_hash.update(history_schedule_identity_sha256);
    schedule_hash.update(terminal_certificate_identity_sha256);
    schedule_hash.update(resident_policy_identity_sha256(policy));
    hash_resident_memory_receipt(&mut schedule_hash, planned_memory);
    hash_resident_host_audit(&mut schedule_hash, host_audit);
    hash_u64(&mut schedule_hash, destinations.len() as u64);
    hash_u64(&mut schedule_hash, consumed.len() as u64);
    let mut destination_buffer_units = 0usize;
    for (index, destination) in destinations.iter().enumerate() {
        deadline.poll(index, "resident terminal destination schedule")?;
        schedule_hash.update(session_nonce_sha256);
        schedule_hash.update(destination.base_domain_identity_sha256);
        schedule_hash.update(destination.logical_domain_identity_sha256);
        hash_u64(&mut schedule_hash, u64::from(destination.slot_index));
        hash_u64(&mut schedule_hash, destination.generation);
        for transfer in destination.family_transfers {
            schedule_hash.update([match transfer {
                GpuBabBoundResidentFamilyTransfer::FreshUpload => 1,
                GpuBabBoundResidentFamilyTransfer::CopyParent => 2,
                GpuBabBoundResidentFamilyTransfer::FreshReplace => 3,
            }]);
        }
        for bytes in destination.layout.family_payload_bytes {
            hash_u64(&mut schedule_hash, bytes as u64);
        }
        hash_u64(
            &mut schedule_hash,
            destination.layout.history_payload_bytes as u64,
        );
        hash_u64(&mut schedule_hash, destination.layout.payload_bytes as u64);
        hash_u64(&mut schedule_hash, destination.history_prefix_bytes as u64);
        hash_u64(&mut schedule_hash, destination.history_suffix_bytes as u64);
        match destination.source {
            None => schedule_hash.update([0]),
            Some(source) => {
                schedule_hash.update([1]);
                schedule_hash.update(source.transcript.session_nonce_sha256);
                schedule_hash.update(source.transcript.logical_domain_identity_sha256);
                hash_u64(&mut schedule_hash, u64::from(source.transcript.slot_index));
                hash_u64(&mut schedule_hash, source.transcript.generation);
                schedule_hash.update([match source.presence {
                    GpuBabBoundResidentSourcePresence::Resident => 1,
                    GpuBabBoundResidentSourcePresence::RefreshOnly => 2,
                }]);
                for bytes in source.family_payload_bytes {
                    hash_u64(&mut schedule_hash, bytes as u64);
                }
                hash_u64(&mut schedule_hash, source.history_payload_bytes as u64);
                hash_u64(&mut schedule_hash, source.resident_device_bytes as u64);
            }
        }
        destination_buffer_units = destination_buffer_units
            .checked_add(
                destination
                    .layout
                    .family_payload_bytes
                    .iter()
                    .filter(|&&bytes| bytes != 0)
                    .count()
                    + usize::from(destination.layout.history_payload_bytes != 0),
            )
            .ok_or_else(|| invalid("resident destination buffer units overflow usize"))?;
    }
    deadline.check("resident terminal destination schedule")?;
    hash_u64(&mut schedule_hash, destination_buffer_units as u64);
    hash_u64(
        &mut schedule_hash,
        planned_transfers.resident_transfer_units as u64,
    );
    hash_u64(
        &mut schedule_hash,
        planned_transfers.resident_host_to_device_transfer_units as u64,
    );
    schedule_hash.update(b"planned-completed-transfer-units\0");
    hash_u64(
        &mut schedule_hash,
        planned_transfers.completed_resident_transfer_units as u64,
    );
    // These zero classes are part of this exact schema, not merely validator
    // defaults. Domain-separate and bind them so a future schema cannot
    // reinterpret the same admission digest as authorizing an uploaded
    // control payload or destination padding.
    schedule_hash.update(b"resident-control-payload-bytes\0");
    hash_u64(&mut schedule_hash, 0);
    schedule_hash.update(b"destination-padding-bytes\0");
    hash_u64(&mut schedule_hash, 0);
    for bytes in planned_transfers
        .fresh_family_host_to_device_bytes
        .into_iter()
        .chain(planned_transfers.replaced_family_host_to_device_bytes)
        .chain(planned_transfers.copied_family_device_to_device_bytes)
        .chain([
            planned_transfers.fresh_history_host_to_device_bytes,
            planned_transfers.delta_history_host_to_device_bytes,
            planned_transfers.history_device_to_device_bytes,
            planned_transfers.resident_host_to_device_bytes,
            planned_transfers.resident_device_to_device_bytes,
        ])
    {
        hash_u64(&mut schedule_hash, bytes as u64);
    }
    for (index, consumed_slot) in consumed.iter().enumerate() {
        deadline.poll(index, "resident terminal consumed schedule")?;
        schedule_hash.update([match consumed_slot.kind {
            GpuBabBoundResidentConsumedKind::Parent => 1,
            GpuBabBoundResidentConsumedKind::Release => 2,
            GpuBabBoundResidentConsumedKind::Evict => 3,
        }]);
        schedule_hash.update(consumed_slot.transcript.session_nonce_sha256);
        schedule_hash.update(consumed_slot.transcript.logical_domain_identity_sha256);
        hash_u64(
            &mut schedule_hash,
            u64::from(consumed_slot.transcript.slot_index),
        );
        hash_u64(&mut schedule_hash, consumed_slot.transcript.generation);
        schedule_hash.update([match consumed_slot.presence {
            GpuBabBoundResidentPresence::Resident => 1,
            GpuBabBoundResidentPresence::RefreshOnly => 2,
        }]);
        for bytes in consumed_slot.source_audit.family_payload_bytes {
            hash_u64(&mut schedule_hash, bytes as u64);
        }
        hash_u64(
            &mut schedule_hash,
            consumed_slot.source_audit.history_payload_bytes as u64,
        );
        hash_u64(&mut schedule_hash, consumed_slot.resident_bytes as u64);
        hash_u64(
            &mut schedule_hash,
            consumed_slot.core_host_charged_bytes as u64,
        );
    }
    deadline.check("resident terminal schedule")?;
    Ok(schedule_hash.finalize().into())
}

impl GpuBabBoundResidentDomainState {
    fn prevalidate_source_authority(
        &mut self,
        request: &GpuBabBoundResidentWaveRequest,
        session_nonce_sha256: [u8; 32],
        deadline: ResidentValidationDeadline,
    ) -> Result<bool> {
        deadline.check("resident source authority")?;
        let mut parent_source_count = 0usize;
        for (group_index, group) in request.parent_groups.iter().enumerate() {
            deadline.poll(group_index, "resident source-authority groups")?;
            if group.source.token().is_some() {
                parent_source_count = parent_source_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("resident parent source count overflows usize"))?;
            }
        }
        deadline.check("resident source-authority group count")?;
        let consumed_count = parent_source_count
            .checked_add(request.release.len())
            .and_then(|value| value.checked_add(request.evict.len()))
            .ok_or_else(|| invalid("resident source authority count overflows usize"))?;
        let maximum_slots = self
            .installed_policy()
            .or(match self.policy_state {
                GpuBabBoundResidentPolicyState::Observed(policy) => Some(policy),
                _ => None,
            })
            .map(|policy| policy.maximum_slots)
            .ok_or_else(|| invalid("resident source authority has no stable policy"))?;
        if consumed_count > maximum_slots {
            return Err(invalid(
                "resident source authority sidecars exceed the fixed slot policy",
            ));
        }
        if self.completed_waves == 0 && consumed_count != 0 {
            return Err(invalid(
                "the first retained-domain wave must be fresh-only with no slot transition",
            ));
        }
        if consumed_count != 0 {
            let required_words = maximum_slots.div_ceil(u64::BITS as usize);
            if self.source_authority_bitmap.len() < required_words {
                return Err(invalid("resident source-authority bitmap is not installed"));
            }
            for chunk in self
                .source_authority_bitmap
                .chunks_mut(VALIDATION_POLL_STRIDE)
            {
                deadline.check("resident source-authority bitmap clear")?;
                chunk.fill(0);
            }
            deadline.check("resident source-authority bitmap clear")?;
        }
        let mut full_refresh_required = false;
        let mut checked_tokens = 0usize;
        for (group_index, group) in request.parent_groups.iter().enumerate() {
            deadline.poll(group_index, "resident source-authority groups")?;
            let Some(token) = group.source.token() else {
                continue;
            };
            checked_tokens += 1;
            let slot_index = usize::try_from(token.slot_index)
                .map_err(|_| invalid("resident source slot index does not fit usize"))?;
            let word = slot_index / u64::BITS as usize;
            let mask = 1u64 << (slot_index % u64::BITS as usize);
            let Some(mark) = self.source_authority_bitmap.get_mut(word) else {
                return Err(invalid("resident source slot index is out of range"));
            };
            if *mark & mask != 0 {
                return Err(invalid(
                    "resident source authority reuses one slot across operation sections",
                ));
            }
            *mark |= mask;
            let (_, live) = self.live_slot_for_token(token, session_nonce_sha256)?;
            let base_group = request.wave.parent_groups.get(group_index).ok_or_else(|| {
                invalid("resident source group has no matching base parent group")
            })?;
            if base_group.parent_identity_sha256 != token.logical_domain_identity_sha256 {
                return Err(invalid(
                    "resident base parent identity does not echo its core source slot",
                ));
            }
            let prefix = split_words(
                group.prefix,
                request.split_history.words.as_ref(),
                "resident source authority prefix",
            )?;
            if prefix.len() != live.snapshot.history.len() {
                return Err(invalid(
                    "resident source prefix differs from its core-owned snapshot",
                ));
            }
            for (candidate, retained) in prefix
                .chunks(VALIDATION_POLL_STRIDE)
                .zip(live.snapshot.history.chunks(VALIDATION_POLL_STRIDE))
            {
                deadline.check("resident source prefix authority")?;
                if candidate != retained {
                    return Err(invalid(
                        "resident source prefix differs from its core-owned snapshot",
                    ));
                }
            }
            if group.source.is_delta() && live.presence == GpuBabBoundResidentPresence::RefreshOnly
            {
                full_refresh_required = true;
            }
        }
        for (kind, tokens) in [
            (GpuBabBoundResidentConsumedKind::Release, &request.release),
            (GpuBabBoundResidentConsumedKind::Evict, &request.evict),
        ] {
            for token in tokens {
                deadline.poll(checked_tokens, "resident source authority")?;
                checked_tokens += 1;
                let slot_index = usize::try_from(token.slot_index)
                    .map_err(|_| invalid("resident source slot index does not fit usize"))?;
                let word = slot_index / u64::BITS as usize;
                let mask = 1u64 << (slot_index % u64::BITS as usize);
                let Some(mark) = self.source_authority_bitmap.get_mut(word) else {
                    return Err(invalid("resident source slot index is out of range"));
                };
                if *mark & mask != 0 {
                    return Err(invalid(
                        "resident source authority reuses one slot across operation sections",
                    ));
                }
                *mark |= mask;
                let (_, live) = self.live_slot_for_token(token, session_nonce_sha256)?;
                if kind == GpuBabBoundResidentConsumedKind::Evict
                    && live.presence != GpuBabBoundResidentPresence::Resident
                {
                    return Err(invalid(
                        "resident eviction requires a physically Resident source",
                    ));
                }
            }
        }
        deadline.check("resident source authority publication")?;
        Ok(full_refresh_required)
    }

    fn live_slot_for_token(
        &self,
        token: &GpuBabBoundResidentSlotRef,
        session_nonce_sha256: [u8; 32],
    ) -> Result<(usize, &GpuBabBoundResidentLiveSlot)> {
        if token.session_nonce_sha256 != session_nonce_sha256 {
            return Err(invalid("resident source token belongs to another session"));
        }
        let slot_index = usize::try_from(token.slot_index)
            .map_err(|_| invalid("resident source slot index does not fit usize"))?;
        let state = self
            .slots
            .get(slot_index)
            .ok_or_else(|| invalid("resident source slot index is out of range"))?;
        let GpuBabBoundResidentSlotState::Live(slot) = state else {
            return Err(invalid(
                "resident source slot is stale, vacant, or poisoned",
            ));
        };
        if slot.generation != token.generation
            || slot.snapshot.logical_domain_identity_sha256 != token.logical_domain_identity_sha256
            || slot.in_flight
        {
            return Err(invalid(
                "resident source token generation/identity is stale or already in flight",
            ));
        }
        Ok((slot_index, slot))
    }

    fn plan_wave(
        &self,
        request: &GpuBabBoundResidentWaveRequest,
        phase: &GpuBabBoundPhaseDescriptor,
        validated_shape: ValidatedWaveShape,
        candidate: GpuBabBoundResidentCandidateSize,
        session_nonce_sha256: [u8; 32],
        open_memory: GpuBabBoundMemoryReceipt,
        host_budget: &mut ResidentHostAdmissionBudget,
        deadline: ResidentValidationDeadline,
    ) -> std::result::Result<GpuBabBoundPendingResidentWave, GpuBabBoundResidentAdmissionError>
    {
        if self.poisoned {
            return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                "retained-domain state is poisoned",
            )));
        }
        let policy = self.installed_policy().ok_or_else(|| {
            GpuBabBoundResidentAdmissionError::Invalid(invalid(
                "retained-domain policy is not installed",
            ))
        })?;
        deadline
            .check("resident materialization plan")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let ledger = self.ledger_audit_with_deadline(Some(deadline))?;
        let retained_before_bytes = ledger.resident_device_bytes;
        let host_before_bytes = ledger.core_host_charged_bytes;
        let history_before_words = ledger.history_words;
        let peak_resident_bytes = retained_before_bytes
            .checked_add(candidate.logical_payload_bytes)
            .ok_or_else(|| invalid("retained transition peak bytes overflow usize"))?;
        let minimum_pending_host_bytes = host_before_bytes
            .checked_add(candidate.host_transition_charge_bytes)
            .ok_or_else(|| invalid("resident pending host bytes overflow usize"))?;
        let pending_history_words = history_before_words
            .checked_add(candidate.history_words)
            .ok_or_else(|| invalid("resident pending history words overflow usize"))?;
        if peak_resident_bytes > policy.maximum_resident_device_bytes
            || minimum_pending_host_bytes > policy.maximum_retained_v2_core_host_charged_bytes
            || pending_history_words > policy.maximum_history_words
        {
            return Err(GpuBabBoundResidentAdmissionError::Decline(
                GpuBabBoundResidentWaveDecline::InsufficientCapacity,
            ));
        }
        let mut vacant_count = 0usize;
        for (index, state) in self.slots.iter().enumerate() {
            deadline
                .poll(index, "resident vacant-slot capacity scan")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            if let GpuBabBoundResidentSlotState::Vacant { high_generation } = state {
                high_generation.checked_add(1).ok_or_else(|| {
                    GpuBabBoundResidentAdmissionError::Poison(invalid(
                        "resident destination generation exhausted",
                    ))
                })?;
                vacant_count += 1;
                if vacant_count == candidate.destination_count {
                    break;
                }
            }
        }
        deadline
            .check("resident vacant-slot capacity scan")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        if vacant_count != candidate.destination_count {
            return Err(GpuBabBoundResidentAdmissionError::Decline(
                GpuBabBoundResidentWaveDecline::InsufficientCapacity,
            ));
        }
        if self.completed_waves == 0 {
            for (group_index, group) in request.parent_groups.iter().enumerate() {
                deadline
                    .poll(group_index, "resident first-wave source mode")
                    .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                if !matches!(
                    &group.source,
                    GpuBabBoundResidentParentSource::FreshUpload { prior: None }
                ) {
                    return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                        "the first retained-domain wave must be fresh-only with no slot transition",
                    )));
                }
            }
            deadline
                .check("resident first-wave source mode")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            if !request.release.is_empty() || !request.evict.is_empty() {
                return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                    "the first retained-domain wave must be fresh-only with no slot transition",
                )));
            }
        }

        let consumed_capacity = request
            .parent_groups
            .len()
            .checked_add(request.release.len())
            .and_then(|value| value.checked_add(request.evict.len()))
            .ok_or_else(|| invalid("resident consumed-slot count overflows usize"))?;
        if consumed_capacity > policy.maximum_slots {
            return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                "resident source/release/eviction sidecars exceed the fixed slot policy",
            )));
        }
        let mut consumed = resident_vec_with_metadata_budget(
            consumed_capacity,
            GPU_BAB_BOUND_HOST_PENDING_SOURCE_METADATA_BYTES,
            "resident consumed source descriptors",
            host_budget,
            deadline,
        )?;
        let mut seen_slots = resident_hash_set_with_metadata_budget(
            consumed_capacity,
            GPU_BAB_BOUND_HOST_PENDING_SOURCE_METADATA_BYTES,
            "resident consumed slot keys",
            host_budget,
            deadline,
        )?;
        let mut group_sources: Vec<Option<(usize, &GpuBabBoundResidentLiveSlot)>> =
            resident_vec_with_metadata_budget(
                request.parent_groups.len(),
                GPU_BAB_BOUND_HOST_PENDING_GROUP_METADATA_BYTES,
                "resident group source descriptors",
                host_budget,
                deadline,
            )?;
        let mut prepared_groups = resident_vec_with_metadata_budget(
            request.parent_groups.len(),
            GPU_BAB_BOUND_HOST_PENDING_GROUP_METADATA_BYTES,
            "resident prepared group descriptors",
            host_budget,
            deadline,
        )?;
        for (group_index, group) in request.parent_groups.iter().enumerate() {
            deadline
                .poll(group_index, "resident prepared source groups")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            let Some(token) = group.source.token() else {
                group_sources.push(None);
                prepared_groups.push(GpuBabBoundPreparedResidentGroup {
                    parent_group_id: group.parent_group_id,
                    prefix: group.prefix,
                    construction: group.construction,
                    source_class: GpuBabBoundResidentSourceClass::FreshUpload,
                    source: None,
                });
                continue;
            };
            if request.wave.parent_groups[group_index].parent_identity_sha256
                != token.logical_domain_identity_sha256
            {
                return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                    format!(
                    "resident group {group_index} base parent identity does not echo its core slot"
                ),
                )));
            }
            let (slot_index, live) = self.live_slot_for_token(token, session_nonce_sha256)?;
            if !seen_slots.insert(slot_index) {
                return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                    "resident source/release/eviction token is duplicated",
                )));
            }
            // The allocation-free source-authority certificate immediately
            // preceding this plan compared the immutable prefix to the exact
            // live snapshot in deadline-polled chunks. Repeating that large
            // equality pass here would add no safe-code mutation defense.
            if group.source.is_delta() && live.presence == GpuBabBoundResidentPresence::RefreshOnly
            {
                return Err(GpuBabBoundResidentAdmissionError::Decline(
                    GpuBabBoundResidentWaveDecline::FullRefreshRequired,
                ));
            }
            let transcript = GpuBabBoundResidentSlotTranscript {
                session_nonce_sha256,
                logical_domain_identity_sha256: live.snapshot.logical_domain_identity_sha256,
                slot_index: slot_index as u32,
                generation: live.generation,
            };
            let source_audit = live.source_audit(transcript);
            consumed.push(GpuBabBoundResidentConsumedSlot {
                slot_index,
                kind: GpuBabBoundResidentConsumedKind::Parent,
                presence: live.presence,
                resident_bytes: if live.presence == GpuBabBoundResidentPresence::Resident {
                    live.layout.payload_bytes
                } else {
                    0
                },
                core_host_charged_bytes: live.layout.core_host_charged_bytes,
                transcript,
                source_audit,
            });
            prepared_groups.push(GpuBabBoundPreparedResidentGroup {
                parent_group_id: group.parent_group_id,
                prefix: group.prefix,
                construction: group.construction,
                source_class: if group.source.is_delta() {
                    GpuBabBoundResidentSourceClass::RetainedDelta
                } else {
                    GpuBabBoundResidentSourceClass::FreshUpload
                },
                source: Some(source_audit),
            });
            group_sources.push(Some((slot_index, live)));
        }
        deadline
            .check("resident prepared source groups")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let mut transition_token_index = 0usize;
        for (kind, tokens) in [
            (GpuBabBoundResidentConsumedKind::Release, &request.release),
            (GpuBabBoundResidentConsumedKind::Evict, &request.evict),
        ] {
            for token in tokens {
                deadline
                    .poll(transition_token_index, "resident release/evict sources")
                    .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                transition_token_index += 1;
                let (slot_index, live) = self.live_slot_for_token(token, session_nonce_sha256)?;
                if kind == GpuBabBoundResidentConsumedKind::Evict
                    && live.presence == GpuBabBoundResidentPresence::RefreshOnly
                {
                    return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                        "explicit eviction requires a physically Resident source",
                    )));
                }
                if !seen_slots.insert(slot_index) {
                    return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                        "resident source/release/eviction token is duplicated",
                    )));
                }
                let transcript = GpuBabBoundResidentSlotTranscript {
                    session_nonce_sha256,
                    logical_domain_identity_sha256: live.snapshot.logical_domain_identity_sha256,
                    slot_index: slot_index as u32,
                    generation: live.generation,
                };
                consumed.push(GpuBabBoundResidentConsumedSlot {
                    slot_index,
                    kind,
                    presence: live.presence,
                    resident_bytes: if live.presence == GpuBabBoundResidentPresence::Resident {
                        live.layout.payload_bytes
                    } else {
                        0
                    },
                    core_host_charged_bytes: live.layout.core_host_charged_bytes,
                    transcript,
                    source_audit: live.source_audit(transcript),
                });
            }
        }
        deadline
            .check("resident release/evict sources")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;

        let planned_transfers =
            expected_resident_transfers_without_copy(request, &group_sources, deadline)?;
        if planned_transfers.resident_transfer_units > candidate.maximum_transfer_units {
            return Err(GpuBabBoundResidentAdmissionError::Poison(invalid(
                "resident transfer plan exceeds its allocation-free unit bound",
            )));
        }
        let minimum_transition_peak = open_memory
            .checked_sum()?
            .checked_add(retained_before_bytes)
            .and_then(|value| value.checked_add(candidate.logical_payload_bytes))
            .and_then(|value| value.checked_add(planned_transfers.resident_host_to_device_bytes))
            .ok_or_else(|| invalid("resident admission minimum peak overflows usize"))?;
        if minimum_transition_peak > request.wave.max_device_bytes {
            return Err(GpuBabBoundResidentAdmissionError::Decline(
                GpuBabBoundResidentWaveDecline::InsufficientCapacity,
            ));
        }

        let destination_count = candidate.destination_count;
        let mut vacant = resident_vec_with_metadata_budget(
            destination_count,
            GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES,
            "resident vacant destination descriptors",
            host_budget,
            deadline,
        )?;
        for (index, state) in self.slots.iter().enumerate() {
            deadline
                .poll(index, "resident vacant destination selection")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            if let GpuBabBoundResidentSlotState::Vacant { high_generation } = state {
                vacant.push((index, *high_generation));
                if vacant.len() == destination_count {
                    break;
                }
            }
        }
        deadline
            .check("resident vacant destination selection")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        if vacant.len() != destination_count {
            return Err(GpuBabBoundResidentAdmissionError::Decline(
                GpuBabBoundResidentWaveDecline::InsufficientCapacity,
            ));
        }
        let (mut release_count, mut evict_count) = (0usize, 0usize);
        for (index, entry) in consumed.iter().enumerate() {
            deadline
                .poll(index, "resident consumed classification")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            match entry.kind {
                GpuBabBoundResidentConsumedKind::Release => release_count += 1,
                GpuBabBoundResidentConsumedKind::Evict => evict_count += 1,
                GpuBabBoundResidentConsumedKind::Parent => {}
            }
        }
        deadline
            .check("resident consumed classification")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let parent_count = consumed
            .len()
            .checked_sub(release_count)
            .and_then(|value| value.checked_sub(evict_count))
            .ok_or_else(|| invalid("resident parent source count underflows"))?;
        let mut destinations = resident_vec_with_metadata_budget(
            destination_count,
            GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES,
            "resident destination plans",
            host_budget,
            deadline,
        )?;
        let mut prepared_release = resident_vec_with_metadata_budget(
            release_count,
            GPU_BAB_BOUND_HOST_PENDING_SOURCE_METADATA_BYTES,
            "resident prepared release descriptors",
            host_budget,
            deadline,
        )?;
        let mut prepared_evict = resident_vec_with_metadata_budget(
            evict_count,
            GPU_BAB_BOUND_HOST_PENDING_SOURCE_METADATA_BYTES,
            "resident prepared eviction descriptors",
            host_budget,
            deadline,
        )?;
        let mut accepted_destinations = resident_vec_with_metadata_budget(
            destination_count,
            GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES,
            "resident accepted destination descriptors",
            host_budget,
            deadline,
        )?;
        let mut destination_tokens = resident_vec_with_metadata_budget(
            destination_count,
            GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES,
            "resident destination tokens",
            host_budget,
            deadline,
        )?;
        let mut evicted_tokens = resident_vec_with_metadata_budget(
            evict_count,
            GPU_BAB_BOUND_HOST_PENDING_SOURCE_METADATA_BYTES,
            "resident evicted slot tokens",
            host_budget,
            deadline,
        )?;
        let snapshots = resident_vec_with_metadata_budget(
            destination_count,
            GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES,
            "resident destination snapshot owners",
            host_budget,
            deadline,
        )?;
        let transfer_prefix_capacity = candidate
            .maximum_transfer_units
            .checked_add(1)
            .ok_or_else(|| invalid("resident transfer-prefix capacity overflows usize"))?;
        let transfer_prefixes = resident_vec_with_metadata_budget(
            transfer_prefix_capacity,
            size_of::<GpuBabBoundResidentTransferReceipt>(),
            "resident transfer-prefix certificates",
            host_budget,
            deadline,
        )?;
        let allocation_prefix_capacity = candidate
            .destination_buffer_units
            .checked_add(1)
            .ok_or_else(|| invalid("resident allocation-prefix capacity overflows usize"))?;
        let allocation_prefixes = resident_vec_with_metadata_budget(
            allocation_prefix_capacity,
            size_of::<GpuBabBoundResidentAllocationPrefix>(),
            "resident allocation-prefix certificates",
            host_budget,
            deadline,
        )?;
        let journal_capacity = consumed_capacity
            .checked_add(destination_count)
            .ok_or_else(|| invalid("resident journal capacity overflows usize"))?;
        let mut journal = resident_vec_with_metadata_budget(
            journal_capacity,
            size_of::<GpuBabBoundResidentJournalEntry>(),
            "resident sealed transition journal",
            host_budget,
            deadline,
        )?;
        for (index, entry) in consumed.iter().enumerate() {
            deadline
                .poll(index, "resident source publication")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            match entry.kind {
                GpuBabBoundResidentConsumedKind::Release => {
                    prepared_release.push(entry.source_audit);
                }
                GpuBabBoundResidentConsumedKind::Evict => {
                    prepared_evict.push(entry.source_audit);
                    evicted_tokens.push(GpuBabBoundResidentSlotRef {
                        session_nonce_sha256,
                        logical_domain_identity_sha256: entry
                            .transcript
                            .logical_domain_identity_sha256,
                        slot_index: entry.transcript.slot_index,
                        generation: entry.transcript.generation,
                    });
                }
                GpuBabBoundResidentConsumedKind::Parent => {}
            }
        }
        deadline
            .check("resident source publication")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        for consumed_index in 0..consumed.len() {
            deadline
                .poll(consumed_index, "resident source journal")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            journal.push(GpuBabBoundResidentJournalEntry::Source { consumed_index });
        }
        for destination_index in 0..destination_count {
            deadline
                .poll(destination_index, "resident destination journal")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            journal.push(GpuBabBoundResidentJournalEntry::Destination { destination_index });
        }
        deadline
            .check("resident sealed transition journal")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let history = validate_resident_history_structure(
            request,
            phase,
            validated_shape.schedule_identity_sha256,
            host_budget,
            deadline,
        )?;
        if history.logical_domain_identities_sha256.len() != destination_count
            || history.base_domain_identities_sha256.len() != destination_count
        {
            return Err(GpuBabBoundResidentAdmissionError::Poison(invalid(
                "resident identity-certificate count differs from the checked sizing prepass",
            )));
        }
        // All overlapping metadata reservations have now been observed and
        // charged. Compact payloads are the final workload-scaled allocations;
        // each family/history reserve is charged while empty before copying.
        let snapshots = materialize_resident_snapshots(
            request,
            &history.logical_domain_identities_sha256,
            snapshots,
            host_budget,
            deadline,
        )?;
        let history_schedule_identity_sha256 = history.schedule_identity_sha256;
        let base_domain_identities_sha256 = history.base_domain_identities_sha256;
        let mut destination_bytes = 0usize;
        let mut destination_core_host_charged_bytes = 0usize;
        let mut destination_history_words = 0usize;
        let mut destination_buffer_units = 0usize;
        let mut group_index = 0usize;
        for (domain_index, (snapshot, base_domain_identity_sha256)) in snapshots
            .iter()
            .zip(base_domain_identities_sha256)
            .enumerate()
        {
            deadline
                .poll(domain_index, "resident destination plans")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            while group_index + 1 < request.wave.parent_groups.len()
                && domain_index
                    >= request.wave.parent_groups[group_index]
                        .first_domain
                        .checked_add(request.wave.parent_groups[group_index].child_cardinality)
                        .ok_or_else(|| invalid("resident group coverage overflows usize"))?
            {
                group_index += 1;
            }
            let resident_group = &request.parent_groups[group_index];
            let source = group_sources[group_index].map(|(slot_index, live)| {
                let transcript = GpuBabBoundResidentSlotTranscript {
                    session_nonce_sha256,
                    logical_domain_identity_sha256: live.snapshot.logical_domain_identity_sha256,
                    slot_index: slot_index as u32,
                    generation: live.generation,
                };
                live.source_audit(transcript)
            });
            let family_transfers = if resident_group.source.is_delta() {
                let (_, live) = group_sources[group_index]
                    .expect("validated retained delta owns a live source");
                let mut transfers = [GpuBabBoundResidentFamilyTransfer::FreshReplace; 6];
                for family in RESIDENT_F32_FAMILIES {
                    if f32_bits_equal_with_deadline(
                        snapshot.family(family),
                        live.snapshot.family(family),
                        deadline,
                    )? {
                        transfers[family.index()] = GpuBabBoundResidentFamilyTransfer::CopyParent;
                    }
                }
                transfers
            } else {
                [GpuBabBoundResidentFamilyTransfer::FreshUpload; 6]
            };
            let layout = GpuBabBoundResidentSlotLayout::from_snapshot(snapshot)?;
            destination_buffer_units = destination_buffer_units
                .checked_add(
                    layout
                        .family_payload_bytes
                        .iter()
                        .filter(|&&bytes| bytes != 0)
                        .count()
                        + usize::from(layout.history_payload_bytes != 0),
                )
                .ok_or_else(|| invalid("resident destination buffer units overflow usize"))?;
            checked_add_to(
                &mut destination_bytes,
                layout.payload_bytes,
                "resident destination byte total",
            )?;
            checked_add_to(
                &mut destination_core_host_charged_bytes,
                layout.core_host_charged_bytes,
                "resident destination core-host charged-byte total",
            )?;
            checked_add_to(
                &mut destination_history_words,
                snapshot.history.len(),
                "resident destination history words",
            )?;
            let (slot_index, high_generation) = vacant[domain_index];
            let generation = high_generation.checked_add(1).ok_or_else(|| {
                GpuBabBoundResidentAdmissionError::Poison(invalid(
                    "resident destination generation exhausted",
                ))
            })?;
            if generation == 0 {
                return Err(GpuBabBoundResidentAdmissionError::Poison(invalid(
                    "resident destination generation wrapped to zero",
                )));
            }
            let view = request.domain_histories[domain_index];
            let prefix_bytes = request.parent_groups[group_index]
                .prefix
                .len
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| invalid("resident prefix bytes overflow usize"))?;
            let suffix_bytes = view
                .suffix
                .len
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| invalid("resident suffix bytes overflow usize"))?;
            destinations.push(GpuBabBoundResidentDestinationPlan {
                slot_index: u32::try_from(slot_index)
                    .map_err(|_| invalid("resident destination slot does not fit u32"))?,
                generation,
                base_domain_identity_sha256,
                logical_domain_identity_sha256: snapshot.logical_domain_identity_sha256,
                layout,
                source,
                family_transfers,
                history_prefix_bytes: prefix_bytes,
                history_suffix_bytes: suffix_bytes,
            });
        }
        deadline
            .check("resident destination plans")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;

        if destination_bytes != candidate.logical_payload_bytes
            || destination_history_words != candidate.history_words
            || destination_buffer_units != candidate.destination_buffer_units
        {
            return Err(GpuBabBoundResidentAdmissionError::Poison(invalid(
                "materialized resident payload differs from the checked sizing prepass",
            )));
        }
        let mut released_on_commit_bytes = 0usize;
        let mut removed_resident = 0usize;
        let mut evicted_resident = 0usize;
        let mut removed_refresh_only = 0usize;
        let mut released_host_payload_bytes = 0usize;
        let mut released_history_words = 0usize;
        for (index, entry) in consumed.iter().enumerate() {
            deadline
                .poll(index, "resident completion accounting")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            checked_add_to(
                &mut released_on_commit_bytes,
                entry.resident_bytes,
                "resident release byte total",
            )?;
            if entry.presence == GpuBabBoundResidentPresence::Resident {
                removed_resident += 1;
                if entry.kind == GpuBabBoundResidentConsumedKind::Evict {
                    evicted_resident += 1;
                }
            } else if entry.kind != GpuBabBoundResidentConsumedKind::Evict {
                removed_refresh_only += 1;
            }
            if entry.kind != GpuBabBoundResidentConsumedKind::Evict {
                checked_add_to(
                    &mut released_host_payload_bytes,
                    entry.core_host_charged_bytes,
                    "resident released host payload",
                )?;
                checked_add_to(
                    &mut released_history_words,
                    entry.source_audit.history_payload_bytes / size_of::<u32>(),
                    "resident released history words",
                )?;
            }
        }
        deadline
            .check("resident completion accounting")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let retained_after_bytes = retained_before_bytes
            .checked_add(destination_bytes)
            .and_then(|value| value.checked_sub(released_on_commit_bytes))
            .ok_or_else(|| invalid("retained completion byte equation under/overflows"))?;
        let (resident_slots_before, refresh_only_slots_before) =
            (ledger.resident_slots, ledger.refresh_only_slots);
        let resident_slots_after = resident_slots_before
            .checked_add(destination_count)
            .and_then(|value| value.checked_sub(removed_resident))
            .ok_or_else(|| invalid("resident slot completion equation under/overflows"))?;
        let refresh_only_slots_after = refresh_only_slots_before
            .checked_sub(removed_refresh_only)
            .and_then(|value| value.checked_add(evicted_resident))
            .ok_or_else(|| invalid("refresh-only slot completion equation under/overflows"))?;

        let mut fresh_domains = 0usize;
        for (group_index, (resident, base)) in request
            .parent_groups
            .iter()
            .zip(request.wave.parent_groups.iter())
            .enumerate()
        {
            deadline
                .poll(group_index, "resident fresh/delta accounting")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            if !resident.source.is_delta() {
                fresh_domains = fresh_domains
                    .checked_add(base.child_cardinality)
                    .ok_or_else(|| invalid("resident fresh-domain count overflows usize"))?;
            }
        }
        deadline
            .check("resident fresh/delta accounting")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let delta_domains = destination_count
            .checked_sub(fresh_domains)
            .ok_or_else(|| invalid("resident fresh/delta domain count underflows"))?;
        for (index, destination) in destinations.iter().enumerate() {
            deadline
                .poll(index, "resident accepted destination publication")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            accepted_destinations.push(GpuBabBoundAcceptedResidentDomain {
                destination: GpuBabBoundResidentSlotTranscript {
                    session_nonce_sha256,
                    logical_domain_identity_sha256: destination.logical_domain_identity_sha256,
                    slot_index: destination.slot_index,
                    generation: destination.generation,
                },
                source: destination.source,
                base_domain_identity_sha256: destination.base_domain_identity_sha256,
                logical_domain_identity_sha256: destination.logical_domain_identity_sha256,
                family_transfers: destination.family_transfers,
                family_payload_bytes: destination.layout.family_payload_bytes,
                history_prefix_bytes: destination.history_prefix_bytes,
                history_suffix_bytes: destination.history_suffix_bytes,
                resident_device_bytes: destination.layout.payload_bytes,
            });
            destination_tokens.push(GpuBabBoundResidentSlotRef {
                session_nonce_sha256,
                logical_domain_identity_sha256: destination.logical_domain_identity_sha256,
                slot_index: destination.slot_index,
                generation: destination.generation,
            });
        }
        deadline
            .check("resident accepted destination publication")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let transfer_prefixes = build_resident_transfer_prefixes(
            &destinations,
            planned_transfers,
            transfer_prefixes,
            deadline,
        )?;
        let allocation_prefixes = build_resident_allocation_prefixes(
            &destinations,
            destination_buffer_units,
            allocation_prefixes,
            deadline,
        )?;
        let terminal_certificate_identity_sha256 = resident_terminal_certificate_identity(
            &transfer_prefixes,
            &allocation_prefixes,
            deadline,
        )?;
        let next_completed_waves = self.completed_waves.checked_add(1).ok_or_else(|| {
            GpuBabBoundResidentAdmissionError::Poison(invalid(
                "retained-domain completed-wave sequence exhausted",
            ))
        })?;
        // Every workload-scaled container was reserved and charged while
        // empty. Snapshot capacities were then charged before each copy. The
        // monotone budget is therefore the conservative transition peak; no
        // late allocator observation can raise it after this point.
        let pending_host_bytes = host_budget.total_charged_bytes();
        let host_after_bytes = host_before_bytes
            .checked_add(destination_core_host_charged_bytes)
            .and_then(|value| value.checked_sub(released_host_payload_bytes))
            .ok_or_else(|| invalid("resident host completion equation under/overflows"))?;
        let history_after_words = history_before_words
            .checked_add(destination_history_words)
            .and_then(|value| value.checked_sub(released_history_words))
            .ok_or_else(|| invalid("resident history completion equation under/overflows"))?;
        let host_audit = GpuBabBoundResidentHostAudit {
            retained_v2_core_host_before_charged_bytes: host_before_bytes,
            retained_v2_core_host_peak_charged_bytes: pending_host_bytes,
            retained_v2_core_host_after_charged_bytes: host_after_bytes,
            history_before_words,
            history_peak_words: pending_history_words,
            history_after_words,
        };
        let planned_memory = GpuBabBoundResidentMemoryReceipt {
            resident_device_before_bytes: retained_before_bytes,
            reserved_destination_bytes: destination_bytes,
            allocated_destination_bytes: destination_bytes,
            released_provisional_destination_bytes: 0,
            planned_release_bytes: released_on_commit_bytes,
            committed_release_bytes: released_on_commit_bytes,
            resident_device_after_bytes: retained_after_bytes,
            resident_queued_upload_bytes: planned_transfers.resident_host_to_device_bytes,
            transition_peak_device_bytes: minimum_transition_peak,
            resident_slots_before,
            refresh_only_slots_before,
            destination_slots: destinations.len(),
            destination_buffer_units,
            allocated_destination_slots: destinations.len(),
            allocated_destination_buffer_units: destination_buffer_units,
            released_provisional_destination_slots: 0,
            released_provisional_destination_buffer_units: 0,
            consumed_parent_slots: parent_count,
            explicitly_released_slots: release_count,
            explicitly_evicted_slots: evict_count,
            resident_slots_after,
            refresh_only_slots_after,
            destination_padding_bytes: 0,
        };
        let schedule_identity_sha256 = resident_terminal_schedule_identity(
            history_schedule_identity_sha256,
            session_nonce_sha256,
            &destinations,
            &consumed,
            planned_transfers,
            planned_memory,
            host_audit,
            terminal_certificate_identity_sha256,
            policy,
            deadline,
        )?;
        let in_flight_slots_before = self.in_flight_slots;
        let reserved_slots_before = self.reserved_slots;
        let in_flight_slots_during = in_flight_slots_before
            .checked_add(consumed.len())
            .ok_or_else(|| invalid("resident accepted in-flight count overflows usize"))?;
        let reserved_slots_during = reserved_slots_before
            .checked_add(destinations.len())
            .ok_or_else(|| invalid("resident accepted reserved count overflows usize"))?;
        let pending = GpuBabBoundPendingResidentWave {
            session_nonce_sha256,
            policy,
            prepared_groups,
            prepared_release,
            prepared_evict,
            destinations,
            destination_snapshots: snapshots,
            accepted_destinations,
            destination_tokens,
            evicted_tokens,
            consumed,
            retained_before_bytes,
            destination_bytes,
            released_on_commit_bytes,
            retained_after_bytes,
            host_audit,
            resident_slots_before,
            refresh_only_slots_before,
            resident_slots_after,
            refresh_only_slots_after,
            fresh_domains,
            delta_domains,
            consumed_parent_slots: parent_count,
            explicitly_released_slots: release_count,
            explicitly_evicted_slots: evict_count,
            destination_buffer_units,
            planned_memory,
            planned_transfers,
            transfer_prefixes,
            allocation_prefixes,
            journal,
            in_flight_slots_before,
            reserved_slots_before,
            in_flight_slots_during,
            reserved_slots_during,
            next_completed_waves,
            schedule_identity_sha256,
        };
        if pending.recompute_transfers_with_deadline(Some(deadline))? != planned_transfers {
            return Err(GpuBabBoundResidentAdmissionError::Poison(invalid(
                "materialized resident transfer plan differs from the no-copy prepass",
            )));
        }
        Ok(pending)
    }

    fn prevalidate_maintenance_authority(
        &self,
        request: &GpuBabBoundResidentMaintenanceRequest,
        maximum_slots: usize,
        session_nonce_sha256: [u8; 32],
        deadline: ResidentValidationDeadline,
    ) -> Result<usize> {
        deadline.check("resident maintenance authority prepass")?;
        let operation_count = request
            .release
            .len()
            .checked_add(request.evict.len())
            .ok_or_else(|| invalid("resident maintenance operation count overflows usize"))?;
        if operation_count == 0 || operation_count > maximum_slots {
            return Err(invalid(
                "resident maintenance operation count is empty or exceeds the slot policy",
            ));
        }
        for (label, tokens) in [("release", &request.release), ("evict", &request.evict)] {
            for (index, pair) in tokens.windows(2).enumerate() {
                deadline.poll(index, "resident maintenance canonical token sections")?;
                if (pair[0].slot_index, pair[0].generation)
                    >= (pair[1].slot_index, pair[1].generation)
                {
                    return Err(invalid(format!(
                        "resident maintenance {label} tokens are not strictly ascending"
                    )));
                }
            }
            deadline.check("resident maintenance canonical token sections")?;
        }
        // Both sections are canonical sorted sequences, so a two-pointer
        // merge detects cross-section reuse without allocating a HashSet.
        let (mut release_index, mut evict_index) = (0usize, 0usize);
        while release_index < request.release.len() && evict_index < request.evict.len() {
            deadline.poll(
                release_index + evict_index,
                "resident maintenance operation disjointness",
            )?;
            match request.release[release_index]
                .slot_index
                .cmp(&request.evict[evict_index].slot_index)
            {
                std::cmp::Ordering::Less => release_index += 1,
                std::cmp::Ordering::Greater => evict_index += 1,
                std::cmp::Ordering::Equal => {
                    return Err(invalid(
                        "resident maintenance slot is duplicated across operation sections",
                    ));
                }
            }
        }
        deadline.check("resident maintenance operation disjointness")?;
        for (authority_index, (kind, token)) in request
            .release
            .iter()
            .map(|token| (GpuBabBoundResidentConsumedKind::Release, token))
            .chain(
                request
                    .evict
                    .iter()
                    .map(|token| (GpuBabBoundResidentConsumedKind::Evict, token)),
            )
            .enumerate()
        {
            deadline.poll(authority_index, "resident maintenance authority prepass")?;
            let (_, live) = self.live_slot_for_token(token, session_nonce_sha256)?;
            if kind == GpuBabBoundResidentConsumedKind::Evict
                && live.presence != GpuBabBoundResidentPresence::Resident
            {
                return Err(invalid(
                    "resident maintenance eviction requires physical Resident presence",
                ));
            }
        }
        deadline.check("resident maintenance authority prepass")?;
        Ok(operation_count)
    }

    fn plan_maintenance(
        &self,
        request: &GpuBabBoundResidentMaintenanceRequest,
        phase: &GpuBabBoundPhaseDescriptor,
        session_nonce_sha256: [u8; 32],
        open_memory: GpuBabBoundMemoryReceipt,
        deadline: ResidentValidationDeadline,
    ) -> std::result::Result<GpuBabBoundPendingResidentMaintenance, GpuBabBoundResidentAdmissionError>
    {
        deadline
            .check("resident maintenance plan")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        if self.poisoned {
            return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                "retained-domain state is poisoned",
            )));
        }
        let policy = self.installed_policy().ok_or_else(|| {
            GpuBabBoundResidentAdmissionError::Invalid(invalid(
                "resident maintenance requires an installed v2 policy",
            ))
        })?;
        if request.deadline > phase.deadline {
            return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                "resident maintenance deadline exceeds its phase deadline",
            )));
        }
        let operation_count = self
            .prevalidate_maintenance_authority(
                request,
                policy.maximum_slots,
                session_nonce_sha256,
                deadline,
            )
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        // Maintenance must remain possible at the retained-v2 host high-water
        // mark. Its four workload-scaled vectors are funded entirely by the
        // configured-slot reserve already present in the live host charge.
        // Seed the budget with the complete nominal plan, then reserve and
        // observe every empty vector before filling any of them.
        let configured_slot_reserve_bytes = self
            .slots
            .capacity()
            .checked_mul(GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES)
            .ok_or_else(|| invalid("maintenance configured-slot reserve overflows usize"))?;
        let slot_table_storage_bytes = self
            .slots
            .capacity()
            .checked_mul(size_of::<GpuBabBoundResidentSlotState>())
            .and_then(|bytes| {
                self.source_authority_bitmap
                    .capacity()
                    .checked_mul(size_of::<u64>())
                    .and_then(|bitmap| bytes.checked_add(bitmap))
            })
            .ok_or_else(|| invalid("maintenance slot-table storage overflows usize"))?;
        let nominal_maintenance_metadata_bytes = operation_count
            .checked_mul(size_of::<GpuBabBoundResidentConsumedSlot>())
            .and_then(|value| {
                request
                    .release
                    .len()
                    .checked_mul(size_of::<GpuBabBoundResidentSourceAudit>())
                    .and_then(|bytes| value.checked_add(bytes))
            })
            .and_then(|value| {
                request
                    .evict
                    .len()
                    .checked_mul(size_of::<GpuBabBoundResidentSourceAudit>())
                    .and_then(|bytes| value.checked_add(bytes))
            })
            .and_then(|value| {
                request
                    .evict
                    .len()
                    .checked_mul(size_of::<GpuBabBoundResidentSlotRef>())
                    .and_then(|bytes| value.checked_add(bytes))
            })
            .and_then(|value| {
                operation_count
                    .checked_mul(GPU_BAB_BOUND_HOST_MAINTENANCE_FIXED_BYTES_PER_OPERATION)
                    .and_then(|bytes| value.checked_add(bytes))
            })
            .ok_or_else(|| invalid("maintenance nominal metadata charge overflows usize"))?;
        let mut maintenance_budget = ResidentHostAdmissionBudget::new(
            configured_slot_reserve_bytes,
            slot_table_storage_bytes,
            nominal_maintenance_metadata_bytes,
            0,
        )
        .map_err(|_| {
            GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                operation_count,
                "resident maintenance configured-slot scratch reserve",
            ))
        })?;
        let mut consumed = resident_vec_with_capacity(
            operation_count,
            "resident maintenance consumed descriptors",
        )?;
        maintenance_budget
            .charge_metadata_capacity(
                operation_count,
                consumed.capacity(),
                size_of::<GpuBabBoundResidentConsumedSlot>(),
            )
            .map_err(|()| {
                GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                    operation_count,
                    "resident maintenance consumed descriptor capacity",
                ))
            })?;
        deadline
            .check("resident maintenance consumed reserve")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let mut release = resident_vec_with_capacity(
            request.release.len(),
            "resident maintenance release descriptors",
        )?;
        maintenance_budget
            .charge_metadata_capacity(
                request.release.len(),
                release.capacity(),
                size_of::<GpuBabBoundResidentSourceAudit>(),
            )
            .map_err(|()| {
                GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                    request.release.len(),
                    "resident maintenance release descriptor capacity",
                ))
            })?;
        deadline
            .check("resident maintenance release reserve")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let mut evict = resident_vec_with_capacity(
            request.evict.len(),
            "resident maintenance eviction descriptors",
        )?;
        maintenance_budget
            .charge_metadata_capacity(
                request.evict.len(),
                evict.capacity(),
                size_of::<GpuBabBoundResidentSourceAudit>(),
            )
            .map_err(|()| {
                GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                    request.evict.len(),
                    "resident maintenance eviction descriptor capacity",
                ))
            })?;
        deadline
            .check("resident maintenance eviction reserve")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let mut evicted_tokens = resident_vec_with_capacity(
            request.evict.len(),
            "resident maintenance eviction tokens",
        )?;
        maintenance_budget
            .charge_metadata_capacity(
                request.evict.len(),
                evicted_tokens.capacity(),
                size_of::<GpuBabBoundResidentSlotRef>(),
            )
            .map_err(|()| {
                GpuBabBoundResidentAdmissionError::Allocation(resident_allocation_error(
                    request.evict.len(),
                    "resident maintenance eviction token capacity",
                ))
            })?;
        deadline
            .check("resident maintenance token reserve")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let mut released_resident_bytes = 0usize;
        let mut released_host_snapshot_bytes = 0usize;
        let mut released_history_words = 0usize;
        let mut released_resident_slots = 0usize;
        let mut released_refresh_slots = 0usize;
        let mut operation_index = 0usize;
        for (kind, tokens) in [
            (GpuBabBoundResidentConsumedKind::Release, &request.release),
            (GpuBabBoundResidentConsumedKind::Evict, &request.evict),
        ] {
            for token in tokens {
                deadline
                    .poll(operation_index, "resident maintenance source audit")
                    .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                operation_index += 1;
                let (slot_index, live) = self.live_slot_for_token(token, session_nonce_sha256)?;
                if kind == GpuBabBoundResidentConsumedKind::Evict
                    && live.presence != GpuBabBoundResidentPresence::Resident
                {
                    return Err(GpuBabBoundResidentAdmissionError::Invalid(invalid(
                        "resident maintenance eviction requires physical Resident presence",
                    )));
                }
                let resident_bytes = if live.presence == GpuBabBoundResidentPresence::Resident {
                    live.layout.payload_bytes
                } else {
                    0
                };
                checked_add_to(
                    &mut released_resident_bytes,
                    resident_bytes,
                    "resident maintenance physical release bytes",
                )?;
                let host_release_bytes = if kind == GpuBabBoundResidentConsumedKind::Release {
                    // The fixed configured-slot reserve remains charged after
                    // Release because the vacant ledger entry and its cleanup
                    // scratch capacity remain allocated for the phase.
                    live.layout.core_host_charged_bytes
                } else {
                    0
                };
                checked_add_to(
                    &mut released_host_snapshot_bytes,
                    host_release_bytes,
                    "resident maintenance host release bytes",
                )?;
                let transcript = GpuBabBoundResidentSlotTranscript {
                    session_nonce_sha256,
                    logical_domain_identity_sha256: live.snapshot.logical_domain_identity_sha256,
                    slot_index: u32::try_from(slot_index)
                        .map_err(|_| invalid("maintenance slot index does not fit u32"))?,
                    generation: live.generation,
                };
                let audit = live.source_audit(transcript);
                if kind == GpuBabBoundResidentConsumedKind::Release {
                    release.push(audit);
                    checked_add_to(
                        &mut released_history_words,
                        audit.history_payload_bytes / size_of::<u32>(),
                        "maintenance released history words",
                    )?;
                    match live.presence {
                        GpuBabBoundResidentPresence::Resident => {
                            released_resident_slots += 1;
                        }
                        GpuBabBoundResidentPresence::RefreshOnly => {
                            released_refresh_slots += 1;
                        }
                    }
                } else {
                    evict.push(audit);
                    evicted_tokens.push(GpuBabBoundResidentSlotRef {
                        session_nonce_sha256,
                        logical_domain_identity_sha256: transcript.logical_domain_identity_sha256,
                        slot_index: transcript.slot_index,
                        generation: transcript.generation,
                    });
                }
                consumed.push(GpuBabBoundResidentConsumedSlot {
                    slot_index,
                    kind,
                    presence: live.presence,
                    resident_bytes,
                    core_host_charged_bytes: live.layout.core_host_charged_bytes,
                    transcript,
                    source_audit: audit,
                });
            }
        }
        deadline
            .check("resident maintenance source audit")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let ledger = self.ledger_audit_with_deadline(Some(deadline))?;
        let retained_before_bytes = ledger.resident_device_bytes;
        let host_before_bytes = ledger.core_host_charged_bytes;
        let history_before_words = ledger.history_words;
        let retained_after_bytes = retained_before_bytes
            .checked_sub(released_resident_bytes)
            .ok_or_else(|| invalid("maintenance retained bytes underflow"))?;
        let (resident_slots_before, refresh_only_slots_before) =
            (ledger.resident_slots, ledger.refresh_only_slots);
        let evicted_slots_count = evict.len();
        let resident_slots_after = resident_slots_before
            .checked_sub(released_resident_slots)
            .and_then(|value| value.checked_sub(evicted_slots_count))
            .ok_or_else(|| invalid("maintenance resident-slot equation underflows"))?;
        let refresh_only_slots_after = refresh_only_slots_before
            .checked_sub(released_refresh_slots)
            .and_then(|value| value.checked_add(evicted_slots_count))
            .ok_or_else(|| invalid("maintenance refresh-slot equation under/overflows"))?;
        let host_after_bytes = host_before_bytes
            .checked_sub(released_host_snapshot_bytes)
            .ok_or_else(|| invalid("maintenance host release underflows live charge"))?;
        let history_after_words = history_before_words
            .checked_sub(released_history_words)
            .ok_or_else(|| invalid("maintenance history release underflows live words"))?;
        let host_audit = GpuBabBoundResidentHostAudit {
            retained_v2_core_host_before_charged_bytes: host_before_bytes,
            retained_v2_core_host_peak_charged_bytes: host_before_bytes,
            retained_v2_core_host_after_charged_bytes: host_after_bytes,
            history_before_words,
            history_peak_words: history_before_words,
            history_after_words,
        };
        let transition_peak_device_bytes = open_memory
            .checked_sum()?
            .checked_add(retained_before_bytes)
            .ok_or_else(|| invalid("resident maintenance peak overflows usize"))?;
        if transition_peak_device_bytes > phase.max_device_bytes {
            return Err(GpuBabBoundResidentAdmissionError::Poison(invalid(
                "resident maintenance peak exceeds the phase device cap",
            )));
        }
        let planned_memory = GpuBabBoundResidentMaintenanceMemoryReceipt {
            resident_device_before_bytes: retained_before_bytes,
            planned_release_device_bytes: released_resident_bytes,
            committed_release_device_bytes: released_resident_bytes,
            resident_device_after_bytes: retained_after_bytes,
            resident_slots_before,
            refresh_only_slots_before,
            released_slots: release.len(),
            evicted_slots: evict.len(),
            resident_slots_after,
            refresh_only_slots_after,
            destination_slots: 0,
            allocated_destination_slots: 0,
            destination_buffer_units: 0,
            allocated_destination_buffer_units: 0,
            allocated_destination_bytes: 0,
            destination_padding_bytes: 0,
            transition_peak_device_bytes,
        };
        let mut schedule_hash = Sha256::new();
        schedule_hash.update(b"ny.gpu-bab-bound.resident-maintenance.v2\0");
        schedule_hash.update(session_nonce_sha256);
        schedule_hash.update(resident_policy_identity_sha256(policy));
        schedule_hash.update(phase.authority.graph_identity_sha256);
        schedule_hash.update(phase.authority.static_phase_identity_sha256);
        hash_resident_maintenance_memory_receipt(&mut schedule_hash, planned_memory);
        hash_resident_host_audit(&mut schedule_hash, host_audit);
        hash_u64(&mut schedule_hash, 0);
        hash_u64(&mut schedule_hash, release.len() as u64);
        hash_u64(&mut schedule_hash, evict.len() as u64);
        for (index, entry) in consumed.iter().enumerate() {
            deadline
                .poll(index, "resident maintenance schedule")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            schedule_hash.update([match entry.kind {
                GpuBabBoundResidentConsumedKind::Release => 1,
                GpuBabBoundResidentConsumedKind::Evict => 2,
                GpuBabBoundResidentConsumedKind::Parent => unreachable!(),
            }]);
            schedule_hash.update(entry.transcript.session_nonce_sha256);
            schedule_hash.update(entry.transcript.logical_domain_identity_sha256);
            hash_u64(&mut schedule_hash, u64::from(entry.transcript.slot_index));
            hash_u64(&mut schedule_hash, entry.transcript.generation);
            schedule_hash.update([match entry.presence {
                GpuBabBoundResidentPresence::Resident => 1,
                GpuBabBoundResidentPresence::RefreshOnly => 2,
            }]);
            for bytes in entry.source_audit.family_payload_bytes {
                hash_u64(&mut schedule_hash, bytes as u64);
            }
            hash_u64(
                &mut schedule_hash,
                entry.source_audit.history_payload_bytes as u64,
            );
            hash_u64(&mut schedule_hash, entry.resident_bytes as u64);
            hash_u64(&mut schedule_hash, entry.core_host_charged_bytes as u64);
        }
        deadline
            .check("resident maintenance schedule")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        hash_u64(&mut schedule_hash, retained_before_bytes as u64);
        hash_u64(&mut schedule_hash, released_resident_bytes as u64);
        hash_u64(&mut schedule_hash, retained_after_bytes as u64);
        hash_u64(&mut schedule_hash, released_host_snapshot_bytes as u64);
        hash_u64(&mut schedule_hash, resident_slots_before as u64);
        hash_u64(&mut schedule_hash, refresh_only_slots_before as u64);
        hash_u64(&mut schedule_hash, resident_slots_after as u64);
        hash_u64(&mut schedule_hash, refresh_only_slots_after as u64);
        let in_flight_slots_before = self.in_flight_slots;
        let reserved_slots_before = self.reserved_slots;
        let in_flight_slots_during = in_flight_slots_before
            .checked_add(consumed.len())
            .ok_or_else(|| invalid("maintenance accepted in-flight count overflows usize"))?;
        let reserved_slots_during = reserved_slots_before;
        Ok(GpuBabBoundPendingResidentMaintenance {
            session_nonce_sha256,
            policy,
            release,
            evict,
            consumed,
            evicted_tokens,
            retained_before_bytes,
            released_resident_bytes,
            retained_after_bytes,
            host_audit,
            resident_slots_before,
            refresh_only_slots_before,
            resident_slots_after,
            refresh_only_slots_after,
            planned_memory,
            in_flight_slots_before,
            reserved_slots_before,
            in_flight_slots_during,
            reserved_slots_during,
            schedule_identity_sha256: schedule_hash.finalize().into(),
        })
    }

    fn reserve_maintenance(
        &mut self,
        pending: &GpuBabBoundPendingResidentMaintenance,
        deadline: Option<ResidentValidationDeadline>,
    ) -> Result<()> {
        if self.poisoned {
            return Err(invalid("cannot reserve maintenance on poisoned state"));
        }
        if self.in_flight_slots != pending.in_flight_slots_before
            || self.reserved_slots != pending.reserved_slots_before
        {
            return Err(invalid(
                "maintenance scalar transaction counters changed before acceptance",
            ));
        }
        if pending.in_flight_slots_during
            != pending
                .in_flight_slots_before
                .checked_add(pending.consumed.len())
                .ok_or_else(|| invalid("maintenance sealed in-flight count overflows usize"))?
            || pending.reserved_slots_during != pending.reserved_slots_before
        {
            return Err(invalid("maintenance sealed counter equation is invalid"));
        }
        if pending.evicted_tokens.len() != pending.evict.len()
            || pending.consumed.len()
                != pending
                    .release
                    .len()
                    .checked_add(pending.evict.len())
                    .ok_or_else(|| invalid("maintenance operation count overflows usize"))?
        {
            return Err(invalid(
                "maintenance sealed publication cardinality is invalid",
            ));
        }
        for (index, consumed) in pending.consumed.iter().enumerate() {
            poll_resident_validation(deadline, index, "maintenance reservation prevalidation")?;
            let (expected_kind, expected_audit) = if index < pending.release.len() {
                (
                    GpuBabBoundResidentConsumedKind::Release,
                    pending.release[index],
                )
            } else {
                let evict_index = index - pending.release.len();
                (
                    GpuBabBoundResidentConsumedKind::Evict,
                    pending.evict[evict_index],
                )
            };
            if consumed.kind != expected_kind || consumed.source_audit != expected_audit {
                return Err(invalid(
                    "maintenance journal sections/audits are not canonical",
                ));
            }
            if expected_kind == GpuBabBoundResidentConsumedKind::Evict {
                let evict_index = index - pending.release.len();
                let token = &pending.evicted_tokens[evict_index];
                if token.session_nonce_sha256 != pending.session_nonce_sha256
                    || token.session_nonce_sha256 != consumed.transcript.session_nonce_sha256
                    || token.slot_index != consumed.transcript.slot_index
                    || token.generation != consumed.transcript.generation
                    || token.logical_domain_identity_sha256
                        != consumed.transcript.logical_domain_identity_sha256
                {
                    return Err(invalid(
                        "maintenance eviction token does not match its canonical journal entry",
                    ));
                }
            }
            let Some(GpuBabBoundResidentSlotState::Live(slot)) =
                self.slots.get(consumed.slot_index)
            else {
                return Err(invalid("maintenance source is no longer live"));
            };
            if slot.in_flight
                || slot.generation != consumed.transcript.generation
                || slot.snapshot.logical_domain_identity_sha256
                    != consumed.transcript.logical_domain_identity_sha256
                || slot.presence != consumed.presence
                || slot.source_audit(consumed.transcript) != consumed.source_audit
            {
                return Err(invalid(
                    "maintenance source identity changed before acceptance",
                ));
            }
        }
        finish_resident_validation(deadline, "maintenance reservation prevalidation")?;
        for consumed in &pending.consumed {
            resident_maybe_inject_journal_panic("maintenance reserve mutation");
            let GpuBabBoundResidentSlotState::Live(slot) = &mut self.slots[consumed.slot_index]
            else {
                unreachable!("maintenance reservation prevalidated every source")
            };
            slot.in_flight = true;
        }
        self.in_flight_slots = pending.in_flight_slots_during;
        self.reserved_slots = pending.reserved_slots_during;
        Ok(())
    }

    fn rollback_maintenance(&mut self, pending: &GpuBabBoundPendingResidentMaintenance) {
        for consumed in &pending.consumed {
            resident_maybe_inject_journal_panic("maintenance rollback mutation");
            let GpuBabBoundResidentSlotState::Live(slot) = &mut self.slots[consumed.slot_index]
            else {
                unreachable!("accepted maintenance journal retains every source")
            };
            slot.in_flight = false;
        }
        self.in_flight_slots = pending.in_flight_slots_before;
        self.reserved_slots = pending.reserved_slots_before;
    }

    fn commit_maintenance(
        &mut self,
        pending: &mut GpuBabBoundPendingResidentMaintenance,
    ) -> Vec<GpuBabBoundResidentSlotRef> {
        for consumed in &pending.consumed {
            resident_maybe_inject_journal_panic("maintenance commit mutation");
            match consumed.kind {
                GpuBabBoundResidentConsumedKind::Release => {
                    self.slots[consumed.slot_index] = GpuBabBoundResidentSlotState::Vacant {
                        high_generation: consumed.transcript.generation,
                    };
                }
                GpuBabBoundResidentConsumedKind::Evict => {
                    let GpuBabBoundResidentSlotState::Live(slot) =
                        &mut self.slots[consumed.slot_index]
                    else {
                        unreachable!("maintenance commit prevalidated eviction source")
                    };
                    slot.presence = GpuBabBoundResidentPresence::RefreshOnly;
                    slot.in_flight = false;
                }
                GpuBabBoundResidentConsumedKind::Parent => {
                    unreachable!("sealed maintenance journal excludes parent sources")
                }
            }
        }
        self.in_flight_slots = pending.in_flight_slots_before;
        self.reserved_slots = pending.reserved_slots_before;
        std::mem::take(&mut pending.evicted_tokens)
    }

    fn reserve_accepted(
        &mut self,
        pending: &GpuBabBoundPendingResidentWave,
        deadline: Option<ResidentValidationDeadline>,
    ) -> Result<()> {
        if self.poisoned {
            return Err(invalid("cannot reserve destinations on poisoned state"));
        }
        if self.in_flight_slots != pending.in_flight_slots_before
            || self.reserved_slots != pending.reserved_slots_before
        {
            return Err(invalid(
                "resident scalar transaction counters changed before acceptance",
            ));
        }
        if pending.journal.len()
            != pending
                .consumed
                .len()
                .checked_add(pending.destinations.len())
                .ok_or_else(|| invalid("resident accepted journal count overflows usize"))?
        {
            return Err(invalid("resident accepted journal is incomplete"));
        }
        if pending.destination_snapshots.len() != pending.destinations.len()
            || pending.accepted_destinations.len() != pending.destinations.len()
            || pending.destination_tokens.len() != pending.destinations.len()
            || pending.evicted_tokens.len() != pending.explicitly_evicted_slots
        {
            return Err(invalid(
                "resident sealed publication cardinality is invalid",
            ));
        }
        if pending.in_flight_slots_during
            != pending
                .in_flight_slots_before
                .checked_add(pending.consumed.len())
                .ok_or_else(|| invalid("resident sealed in-flight count overflows usize"))?
            || pending.reserved_slots_during
                != pending
                    .reserved_slots_before
                    .checked_add(pending.destinations.len())
                    .ok_or_else(|| invalid("resident sealed reserved count overflows usize"))?
        {
            return Err(invalid("resident sealed counter equation is invalid"));
        }
        let source_count = pending.consumed.len();
        let mut evicted_token_index = 0usize;
        for (position, entry) in pending.journal.iter().enumerate() {
            poll_resident_validation(deadline, position, "resident reservation prevalidation")?;
            if position < source_count {
                if *entry
                    != (GpuBabBoundResidentJournalEntry::Source {
                        consumed_index: position,
                    })
                {
                    return Err(invalid("resident source journal is not canonical"));
                }
                let consumed = &pending.consumed[position];
                let Some(GpuBabBoundResidentSlotState::Live(slot)) =
                    self.slots.get(consumed.slot_index)
                else {
                    return Err(invalid("accepted source slot changed before reservation"));
                };
                if slot.in_flight
                    || slot.generation != consumed.transcript.generation
                    || slot.snapshot.logical_domain_identity_sha256
                        != consumed.transcript.logical_domain_identity_sha256
                    || slot.presence != consumed.presence
                    || slot.source_audit(consumed.transcript) != consumed.source_audit
                {
                    return Err(invalid(
                        "accepted source identity/presence/layout changed before reservation",
                    ));
                }
                if consumed.kind == GpuBabBoundResidentConsumedKind::Evict {
                    let token = pending
                        .evicted_tokens
                        .get(evicted_token_index)
                        .ok_or_else(|| invalid("resident eviction token journal is incomplete"))?;
                    if token.session_nonce_sha256 != pending.session_nonce_sha256
                        || token.session_nonce_sha256 != consumed.transcript.session_nonce_sha256
                        || token.slot_index != consumed.transcript.slot_index
                        || token.generation != consumed.transcript.generation
                        || token.logical_domain_identity_sha256
                            != consumed.transcript.logical_domain_identity_sha256
                    {
                        return Err(invalid(
                            "resident eviction token does not match its canonical journal entry",
                        ));
                    }
                    evicted_token_index += 1;
                }
            } else {
                let destination_index = position - source_count;
                if *entry != (GpuBabBoundResidentJournalEntry::Destination { destination_index }) {
                    return Err(invalid("resident destination journal is not canonical"));
                }
                let destination = &pending.destinations[destination_index];
                let index = destination.slot_index as usize;
                let Some(GpuBabBoundResidentSlotState::Vacant { high_generation }) =
                    self.slots.get(index)
                else {
                    return Err(invalid(
                        "accepted destination slot changed before generation burn",
                    ));
                };
                let token = &pending.destination_tokens[destination_index];
                let accepted = &pending.accepted_destinations[destination_index].destination;
                if high_generation.checked_add(1) != Some(destination.generation)
                    || token.session_nonce_sha256 != pending.session_nonce_sha256
                    || accepted.session_nonce_sha256 != pending.session_nonce_sha256
                    || token.slot_index != destination.slot_index
                    || token.generation != destination.generation
                    || token.logical_domain_identity_sha256
                        != destination.logical_domain_identity_sha256
                    || accepted.slot_index != destination.slot_index
                    || accepted.generation != destination.generation
                    || accepted.logical_domain_identity_sha256
                        != destination.logical_domain_identity_sha256
                {
                    return Err(invalid(
                        "accepted destination generation/token seal is invalid",
                    ));
                }
            }
        }
        if evicted_token_index != pending.evicted_tokens.len() {
            return Err(invalid(
                "resident eviction-token journal has trailing entries",
            ));
        }
        finish_resident_validation(deadline, "resident reservation prevalidation")?;
        for entry in &pending.journal {
            resident_maybe_inject_journal_panic("resident reserve mutation");
            match *entry {
                GpuBabBoundResidentJournalEntry::Source { consumed_index } => {
                    let consumed = &pending.consumed[consumed_index];
                    let GpuBabBoundResidentSlotState::Live(slot) =
                        &mut self.slots[consumed.slot_index]
                    else {
                        unreachable!("reservation prevalidated every journal source")
                    };
                    slot.in_flight = true;
                }
                GpuBabBoundResidentJournalEntry::Destination { destination_index } => {
                    let destination = &pending.destinations[destination_index];
                    self.slots[destination.slot_index as usize] =
                        GpuBabBoundResidentSlotState::Reserved {
                            generation: destination.generation,
                            layout: destination.layout,
                        };
                }
            }
        }
        self.in_flight_slots = pending.in_flight_slots_during;
        self.reserved_slots = pending.reserved_slots_during;
        Ok(())
    }

    /// Roll back only a core-known zero-work or exactly-receipted failure.
    /// Destination generations remain burned; source tokens are never returned
    /// after acceptance, and the caller immediately poisons the phase.
    fn rollback_accepted(&mut self, pending: &GpuBabBoundPendingResidentWave) {
        for entry in &pending.journal {
            resident_maybe_inject_journal_panic("resident rollback mutation");
            match *entry {
                GpuBabBoundResidentJournalEntry::Source { consumed_index } => {
                    let consumed = &pending.consumed[consumed_index];
                    let GpuBabBoundResidentSlotState::Live(slot) =
                        &mut self.slots[consumed.slot_index]
                    else {
                        unreachable!("accepted rollback journal retains every source")
                    };
                    slot.in_flight = false;
                }
                GpuBabBoundResidentJournalEntry::Destination { destination_index } => {
                    let destination = &pending.destinations[destination_index];
                    self.slots[destination.slot_index as usize] =
                        GpuBabBoundResidentSlotState::Vacant {
                            high_generation: destination.generation,
                        };
                }
            }
        }
        self.in_flight_slots = pending.in_flight_slots_before;
        self.reserved_slots = pending.reserved_slots_before;
    }

    fn commit_completed(
        &mut self,
        pending: &mut GpuBabBoundPendingResidentWave,
    ) -> GpuBabBoundResidentCommit {
        let destination_snapshots = std::mem::take(&mut pending.destination_snapshots);
        let source_count = pending.consumed.len();
        let (source_journal, destination_journal) = pending.journal.split_at(source_count);
        for entry in source_journal {
            resident_maybe_inject_journal_panic("resident commit mutation");
            match *entry {
                GpuBabBoundResidentJournalEntry::Source { consumed_index } => {
                    let consumed = &pending.consumed[consumed_index];
                    match consumed.kind {
                        GpuBabBoundResidentConsumedKind::Parent
                        | GpuBabBoundResidentConsumedKind::Release => {
                            self.slots[consumed.slot_index] =
                                GpuBabBoundResidentSlotState::Vacant {
                                    high_generation: consumed.transcript.generation,
                                };
                        }
                        GpuBabBoundResidentConsumedKind::Evict => {
                            let GpuBabBoundResidentSlotState::Live(slot) =
                                &mut self.slots[consumed.slot_index]
                            else {
                                unreachable!("accepted commit journal retains every eviction")
                            };
                            slot.presence = GpuBabBoundResidentPresence::RefreshOnly;
                            slot.in_flight = false;
                        }
                    }
                }
                GpuBabBoundResidentJournalEntry::Destination { .. } => {
                    unreachable!("sealed source journal contains only sources")
                }
            }
        }
        for (entry, snapshot) in destination_journal.iter().zip(destination_snapshots) {
            resident_maybe_inject_journal_panic("resident commit mutation");
            let GpuBabBoundResidentJournalEntry::Destination { destination_index } = *entry else {
                unreachable!("sealed destination journal contains only destinations")
            };
            let destination = &pending.destinations[destination_index];
            self.slots[destination.slot_index as usize] =
                GpuBabBoundResidentSlotState::Live(GpuBabBoundResidentLiveSlot {
                    generation: destination.generation,
                    snapshot,
                    layout: destination.layout,
                    presence: GpuBabBoundResidentPresence::Resident,
                    in_flight: false,
                });
        }
        self.in_flight_slots = pending.in_flight_slots_before;
        self.reserved_slots = pending.reserved_slots_before;
        self.completed_waves = pending.next_completed_waves;
        GpuBabBoundResidentCommit {
            destination_tokens: std::mem::take(&mut pending.destination_tokens),
            evicted_tokens: std::mem::take(&mut pending.evicted_tokens),
        }
    }
}

struct GpuBabBoundResidentCommit {
    destination_tokens: Vec<GpuBabBoundResidentSlotRef>,
    evicted_tokens: Vec<GpuBabBoundResidentSlotRef>,
}

fn guarded_rollback_resident(
    state: &mut GpuBabBoundResidentDomainState,
    pending: &GpuBabBoundPendingResidentWave,
) -> bool {
    catch_tcb_unwind(|| state.rollback_accepted(pending)).is_ok()
}

fn guarded_commit_resident(
    state: &mut GpuBabBoundResidentDomainState,
    pending: &mut GpuBabBoundPendingResidentWave,
) -> std::result::Result<GpuBabBoundResidentCommit, ()> {
    catch_tcb_unwind(|| state.commit_completed(pending))
}

fn guarded_rollback_maintenance(
    state: &mut GpuBabBoundResidentDomainState,
    pending: &GpuBabBoundPendingResidentMaintenance,
) -> bool {
    catch_tcb_unwind(|| state.rollback_maintenance(pending)).is_ok()
}

fn guarded_commit_maintenance(
    state: &mut GpuBabBoundResidentDomainState,
    pending: &mut GpuBabBoundPendingResidentMaintenance,
) -> std::result::Result<Vec<GpuBabBoundResidentSlotRef>, ()> {
    catch_tcb_unwind(|| state.commit_maintenance(pending))
}

#[derive(Debug)]
struct ValidatedResidentHistory {
    logical_domain_identities_sha256: Vec<[u8; 32]>,
    base_domain_identities_sha256: Vec<[u8; 32]>,
    schedule_identity_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuBabBoundResidentCandidateSize {
    logical_payload_bytes: usize,
    host_metadata_charge_bytes: usize,
    host_transition_charge_bytes: usize,
    history_words: usize,
    destination_count: usize,
    destination_buffer_units: usize,
    maximum_transfer_units: usize,
}

/// Compute the exact compact payload before copying a single snapshot value.
/// Shared prefixes and overlapping operand views are deliberately counted once
/// per destination slot because each committed slot owns an independent full
/// logical payload. This function performs no candidate `Vec`/`Arc` allocation.
fn resident_candidate_size(
    request: &GpuBabBoundResidentWaveRequest,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<GpuBabBoundResidentCandidateSize> {
    if request.parent_groups.len() != request.wave.parent_groups.len()
        || request.domain_histories.len() != request.wave.domains.len()
        || request.split_history.words.len() > GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS
    {
        return Err(invalid(
            "resident candidate sidecars or split-history arena size are invalid",
        ));
    }
    for (label, tokens) in [("release", &request.release), ("evict", &request.evict)] {
        for (index, pair) in tokens.windows(2).enumerate() {
            poll_resident_validation(deadline, index, "resident canonical source sections")?;
            if (pair[0].slot_index, pair[0].generation) >= (pair[1].slot_index, pair[1].generation)
            {
                return Err(invalid(format!(
                    "resident wave {label} tokens are not strictly ascending"
                )));
            }
        }
        finish_resident_validation(deadline, "resident canonical source sections")?;
    }
    let (mut release_index, mut evict_index) = (0usize, 0usize);
    while release_index < request.release.len() && evict_index < request.evict.len() {
        poll_resident_validation(
            deadline,
            release_index + evict_index,
            "resident source-section disjointness",
        )?;
        match request.release[release_index]
            .slot_index
            .cmp(&request.evict[evict_index].slot_index)
        {
            std::cmp::Ordering::Less => release_index += 1,
            std::cmp::Ordering::Greater => evict_index += 1,
            std::cmp::Ordering::Equal => {
                return Err(invalid(
                    "resident wave release and evict sections reuse one slot",
                ));
            }
        }
    }
    finish_resident_validation(deadline, "resident source-section disjointness")?;
    let mut logical_payload_bytes = 0usize;
    let mut history_words = 0usize;
    let mut cursor = 0usize;
    let mut destination_count = 0usize;
    let mut destination_buffer_units = 0usize;
    let mut maximum_transfer_units = 0usize;
    for (group_index, (base_group, resident_group)) in request
        .wave
        .parent_groups
        .iter()
        .zip(request.parent_groups.iter())
        .enumerate()
    {
        poll_resident_validation(deadline, group_index, "resident candidate groups")?;
        if resident_group.parent_group_id != base_group.parent_group_id
            || resident_group.prefix.start != cursor
            || !resident_group
                .prefix
                .start
                .is_multiple_of(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
        {
            return Err(invalid(
                "resident candidate prefix is not in exact canonical group-major order",
            ));
        }
        let prefix = split_words(
            resident_group.prefix,
            request.split_history.words.as_ref(),
            "resident candidate prefix",
        )?;
        cursor = resident_group.prefix.checked_end(
            request.split_history.words.len(),
            "resident candidate prefix",
        )?;
        let end_domain = base_group
            .first_domain
            .checked_add(base_group.child_cardinality)
            .ok_or_else(|| invalid("resident candidate group coverage overflows usize"))?;
        for domain_index in base_group.first_domain..end_domain {
            poll_resident_validation(deadline, domain_index, "resident candidate domains")?;
            destination_count = destination_count
                .checked_add(1)
                .ok_or_else(|| invalid("resident candidate domain count overflows usize"))?;
            let domain = request
                .wave
                .domains
                .get(domain_index)
                .ok_or_else(|| invalid("resident candidate domain is out of range"))?;
            let view = request
                .domain_histories
                .get(domain_index)
                .ok_or_else(|| invalid("resident candidate history view is missing"))?;
            if view.suffix.start != cursor
                || !view
                    .suffix
                    .start
                    .is_multiple_of(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
            {
                return Err(invalid(
                    "resident candidate suffix is not in exact canonical group-major order",
                ));
            }
            let suffix = split_words(
                view.suffix,
                request.split_history.words.as_ref(),
                "resident candidate suffix",
            )?;
            cursor = view.suffix.checked_end(
                request.split_history.words.len(),
                "resident candidate suffix",
            )?;
            let domain_history_words = prefix
                .len()
                .checked_add(suffix.len())
                .ok_or_else(|| invalid("resident candidate history length overflows usize"))?;
            history_words = history_words
                .checked_add(domain_history_words)
                .ok_or_else(|| invalid("resident candidate history total overflows usize"))?;
            let history_bytes = domain_history_words
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| invalid("resident candidate history bytes overflow usize"))?;
            if history_bytes != 0 {
                destination_buffer_units = destination_buffer_units
                    .checked_add(1)
                    .ok_or_else(|| invalid("resident candidate buffer units overflow usize"))?;
            }
            maximum_transfer_units = maximum_transfer_units
                .checked_add(usize::from(!prefix.is_empty()))
                .and_then(|value| value.checked_add(usize::from(!suffix.is_empty())))
                .ok_or_else(|| invalid("resident candidate transfer units overflow usize"))?;
            checked_add_to(
                &mut logical_payload_bytes,
                history_bytes,
                "resident candidate host bytes",
            )?;
            for (range, arena_len, label) in [
                (
                    domain.operands.activation,
                    request.wave.domain_arena.activation.len(),
                    "resident candidate activation",
                ),
                (
                    domain.operands.beta,
                    request.wave.domain_arena.beta.len(),
                    "resident candidate beta",
                ),
                (
                    domain.operands.abs,
                    request.wave.domain_arena.abs.len(),
                    "resident candidate abs",
                ),
                (
                    domain.operands.box_lower,
                    request.wave.domain_arena.box_lower.len(),
                    "resident candidate box lower",
                ),
                (
                    domain.operands.box_upper,
                    request.wave.domain_arena.box_upper.len(),
                    "resident candidate box upper",
                ),
                (
                    domain.operands.cached_la,
                    request.wave.domain_arena.cached_la.len(),
                    "resident candidate cached-lA",
                ),
            ] {
                range.checked_end(arena_len, label)?;
                let bytes = range
                    .len
                    .checked_mul(size_of::<f32>())
                    .ok_or_else(|| invalid(format!("{label} bytes overflow usize")))?;
                if bytes != 0 {
                    destination_buffer_units = destination_buffer_units
                        .checked_add(1)
                        .ok_or_else(|| invalid("resident candidate buffer units overflow usize"))?;
                    maximum_transfer_units =
                        maximum_transfer_units.checked_add(1).ok_or_else(|| {
                            invalid("resident candidate transfer units overflow usize")
                        })?;
                }
                checked_add_to(
                    &mut logical_payload_bytes,
                    bytes,
                    "resident candidate host bytes",
                )?;
            }
        }
    }
    finish_resident_validation(deadline, "resident candidate sizing")?;
    if destination_count != request.wave.domains.len()
        || cursor != request.split_history.words.len()
    {
        return Err(invalid(
            "resident candidate views do not exactly cover domains and split-history arena",
        ));
    }
    let consumed_count = request
        .parent_groups
        .len()
        .checked_add(request.release.len())
        .and_then(|value| value.checked_add(request.evict.len()))
        .ok_or_else(|| invalid("resident candidate source metadata count overflows usize"))?;
    let history_records = history_words / GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS;
    // Conservative requested-capacity multiplicities for every core-owned
    // workload-scaled container that may overlap admission. These are an
    // allocation-free first gate; allocator-observed capacities are charged
    // again after each fallible stage and may only increase the final charge.
    let domain_container_entries = destination_count
        .checked_mul(5)
        .ok_or_else(|| invalid("resident domain-container multiplicity overflows usize"))?;
    let group_container_entries = request
        .parent_groups
        .len()
        .checked_mul(2)
        .ok_or_else(|| invalid("resident group-container multiplicity overflows usize"))?;
    let release_count = request.release.len();
    let evict_count = request.evict.len();
    let source_container_entries = consumed_count
        .checked_mul(2)
        .and_then(|value| value.checked_add(release_count))
        .and_then(|value| value.checked_add(evict_count.checked_mul(2)?))
        .ok_or_else(|| invalid("resident source-container multiplicity overflows usize"))?;
    let base_validator_entries = request
        .parent_groups
        .len()
        .checked_add(destination_count.checked_mul(2).ok_or_else(|| {
            invalid("resident base-validator domain multiplicity overflows usize")
        })?)
        .ok_or_else(|| invalid("resident base-validator multiplicity overflows usize"))?;
    // Prefix keys are held once/group; each suffix can overlap a key table, a
    // record table, and the group's bounded sibling plan. Three times the
    // already prefix-amplified logical history record count safely dominates.
    let history_validation_entries = history_records
        .checked_mul(3)
        .and_then(|value| value.checked_add(destination_count.checked_mul(3)?))
        .ok_or_else(|| invalid("resident history-validator multiplicity overflows usize"))?;
    let journal_entries = consumed_count
        .checked_add(destination_count)
        .ok_or_else(|| invalid("resident journal entry count overflows usize"))?;
    let terminal_certificate_bytes = maximum_transfer_units
        .checked_add(1)
        .and_then(|entries| entries.checked_mul(size_of::<GpuBabBoundResidentTransferReceipt>()))
        .and_then(|transfer_bytes| {
            destination_buffer_units
                .checked_add(1)
                .and_then(|entries| {
                    entries.checked_mul(size_of::<GpuBabBoundResidentAllocationPrefix>())
                })
                .and_then(|allocation_bytes| transfer_bytes.checked_add(allocation_bytes))
        })
        .and_then(|value| {
            journal_entries
                .checked_mul(size_of::<GpuBabBoundResidentJournalEntry>())
                .and_then(|journal_bytes| value.checked_add(journal_bytes))
        })
        .ok_or_else(|| invalid("resident terminal-certificate charge overflows usize"))?;
    let metadata_bytes = domain_container_entries
        .checked_mul(GPU_BAB_BOUND_HOST_PENDING_DOMAIN_METADATA_BYTES)
        .and_then(|value| {
            group_container_entries
                .checked_mul(GPU_BAB_BOUND_HOST_PENDING_GROUP_METADATA_BYTES)
                .and_then(|groups| value.checked_add(groups))
        })
        .and_then(|value| {
            source_container_entries
                .checked_mul(GPU_BAB_BOUND_HOST_PENDING_SOURCE_METADATA_BYTES)
                .and_then(|sources| value.checked_add(sources))
        })
        .and_then(|value| {
            base_validator_entries
                .checked_add(history_validation_entries)?
                .checked_mul(GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES)
                .and_then(|records| value.checked_add(records))
        })
        .and_then(|value| value.checked_add(terminal_certificate_bytes))
        .ok_or_else(|| invalid("resident candidate metadata charge overflows usize"))?;
    let host_transition_charge_bytes = logical_payload_bytes
        .checked_add(metadata_bytes)
        .ok_or_else(|| invalid("resident candidate host transition charge overflows usize"))?;
    Ok(GpuBabBoundResidentCandidateSize {
        logical_payload_bytes,
        host_metadata_charge_bytes: metadata_bytes,
        host_transition_charge_bytes,
        history_words,
        destination_count,
        destination_buffer_units,
        maximum_transfer_units,
    })
}

fn hash_resident_logical_domain(
    phase: &GpuBabBoundPhaseDescriptor,
    history_prefix: &[u32],
    history_suffix: &[u32],
    families: &[&[f32]; 6],
    deadline: ResidentValidationDeadline,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.resident-logical-domain.v2\0");
    hash.update(phase.authority.graph_identity_sha256);
    hash.update(phase.authority.static_phase_identity_sha256);
    hash.update(phase.authority.input_identity_sha256);
    hash.update(phase.authority.root_bounds_identity_sha256);
    hash.update(phase.authority.relaxation_identity_sha256);
    hash.update(phase.authority.objective_set_identity_sha256);
    let history_words = history_prefix
        .len()
        .checked_add(history_suffix.len())
        .ok_or_else(|| invalid("logical child history word count overflows usize"))?;
    hash_u64(&mut hash, history_words as u64);
    for (index, word) in history_prefix.iter().chain(history_suffix).enumerate() {
        deadline.poll(index, "resident logical history identity")?;
        hash.update(word.to_le_bytes());
    }
    deadline.check("resident logical history identity")?;
    for family in families {
        hash_f32s_into_with_deadline(
            &mut hash,
            family,
            Some(deadline),
            "resident logical family identity",
        )?;
    }
    deadline.check("resident logical-domain identity")?;
    Ok(hash.finalize().into())
}

fn validate_resident_history_structure(
    request: &GpuBabBoundResidentWaveRequest,
    phase: &GpuBabBoundPhaseDescriptor,
    base_wave_schedule_identity_sha256: [u8; 32],
    budget: &mut ResidentHostAdmissionBudget,
    deadline: ResidentValidationDeadline,
) -> std::result::Result<ValidatedResidentHistory, GpuBabBoundResidentAdmissionError> {
    let words = request.split_history.words.as_ref();
    if words.len() > GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS {
        return Err(invalid_admission(
            "split-history arena exceeds the finite word cap",
        ));
    }
    if request.parent_groups.len() != request.wave.parent_groups.len()
        || request.domain_histories.len() != request.wave.domains.len()
    {
        return Err(invalid_admission(
            "resident parent/history sidecars must exactly match groups/domains",
        ));
    }
    let mut cursor = 0usize;
    let mut logical_identities = resident_hash_set_with_metadata_budget(
        request.wave.domains.len(),
        GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
        "resident logical identities",
        budget,
        deadline,
    )?;
    let mut logical_domain_identities_sha256 = resident_vec_with_metadata_budget(
        request.wave.domains.len(),
        GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
        "resident certified logical-domain identities",
        budget,
        deadline,
    )?;
    let mut base_domain_identities_sha256 = resident_vec_with_metadata_budget(
        request.wave.domains.len(),
        GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
        "resident certified base-domain identities",
        budget,
        deadline,
    )?;
    let mut schedule_hash = Sha256::new();
    schedule_hash.update(b"ny.gpu-bab-bound.resident-schedule.v2\0");
    schedule_hash.update(base_wave_schedule_identity_sha256);
    hash_u64(&mut schedule_hash, request.parent_groups.len() as u64);
    hash_u64(&mut schedule_hash, request.wave.domains.len() as u64);
    hash_u64(&mut schedule_hash, words.len() as u64);
    for (group_index, (base_group, resident_group)) in request
        .wave
        .parent_groups
        .iter()
        .zip(request.parent_groups.iter())
        .enumerate()
    {
        deadline
            .poll(group_index, "resident split-history groups")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        if resident_group.parent_group_id != base_group.parent_group_id
            || resident_group.prefix.start != cursor
        {
            return Err(invalid_admission(format!(
                "resident group {group_index} does not exactly echo group ID/canonical prefix start"
            )));
        }
        let prefix = split_words(resident_group.prefix, words, "resident parent prefix")?;
        cursor = resident_group
            .prefix
            .checked_end(words.len(), "resident parent prefix")?;
        hash_u64(&mut schedule_hash, resident_group.parent_group_id);
        hash_u64(&mut schedule_hash, prefix.len() as u64);
        for (word_index, word) in prefix.iter().enumerate() {
            deadline
                .poll(word_index, "resident split-history prefix identity")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            schedule_hash.update(word.to_le_bytes());
        }
        deadline
            .check("resident split-history prefix identity")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let prefix_keys =
            validate_literal_prefix(prefix, "resident parent prefix", budget, deadline)?;
        schedule_hash.update([u8::from(resident_group.source.is_delta())]);
        schedule_hash.update([match resident_group.construction {
            GpuBabBoundResidentConstruction::AppendReluChildren => 1,
            GpuBabBoundResidentConstruction::FreshReplace => 2,
        }]);
        match resident_group.source.token() {
            None => schedule_hash.update([0]),
            Some(token) => {
                schedule_hash.update([1]);
                schedule_hash.update(token.session_nonce_sha256);
                schedule_hash.update(token.logical_domain_identity_sha256);
                hash_u64(&mut schedule_hash, u64::from(token.slot_index));
                hash_u64(&mut schedule_hash, token.generation);
            }
        }
        let end_domain = base_group
            .first_domain
            .checked_add(base_group.child_cardinality)
            .ok_or_else(|| invalid("resident group domain coverage overflows usize"))?;
        let mut sibling_plan: Option<Vec<(SplitLiteralKey, u32)>> = None;
        let mut last_sibling_pattern = None;
        let mut full_cardinality_for_group = None;
        for domain_index in base_group.first_domain..end_domain {
            deadline
                .poll(domain_index, "resident split-history domains")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            let view = request.domain_histories[domain_index];
            if view.suffix.start != cursor {
                return Err(invalid_admission(format!(
                    "domain {domain_index} split suffix is not in canonical group-major order"
                )));
            }
            let suffix = split_words(view.suffix, words, "resident child suffix")?;
            cursor = view
                .suffix
                .checked_end(words.len(), "resident child suffix")?;
            let suffix_records_count = suffix.len() / GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS;
            match resident_group.construction {
                GpuBabBoundResidentConstruction::AppendReluChildren => {
                    if suffix_records_count == 0
                        || suffix_records_count > GPU_BAB_BOUND_MAX_APPEND_SPLITS
                    {
                        return Err(invalid_admission(format!(
                            "append domain {domain_index} suffix count must be in 1..={GPU_BAB_BOUND_MAX_APPEND_SPLITS}"
                        )));
                    }
                    let full_cardinality = 1usize
                        .checked_shl(suffix_records_count as u32)
                        .ok_or_else(|| invalid("append truth-table cardinality overflows usize"))?;
                    if base_group.child_cardinality > full_cardinality {
                        return Err(invalid_admission(format!(
                            "append group {group_index} has more children than its truth table"
                        )));
                    }
                    full_cardinality_for_group = Some(full_cardinality);
                }
                GpuBabBoundResidentConstruction::FreshReplace => {
                    if resident_group.source.is_delta() {
                        return Err(invalid_admission(
                            "FreshReplace construction requires the explicit FreshUpload source",
                        ));
                    }
                    if !suffix.is_empty() || view.branch_pattern != 0 {
                        return Err(invalid_admission(format!(
                            "fresh-replace domain {domain_index} must preserve history with an empty zero-pattern suffix"
                        )));
                    }
                }
            }
            let (suffix_records, derived_pattern) = validate_literal_suffix(
                &prefix_keys,
                suffix,
                "resident child suffix",
                budget,
                deadline,
            )?;
            if view.branch_pattern != derived_pattern
                || (resident_group.construction
                    == GpuBabBoundResidentConstruction::AppendReluChildren
                    && last_sibling_pattern.is_some_and(|previous| derived_pattern <= previous))
            {
                return Err(invalid_admission(format!(
                    "domain {domain_index} branch pattern is false or not strictly increasing"
                )));
            }
            if resident_group.construction == GpuBabBoundResidentConstruction::AppendReluChildren {
                last_sibling_pattern = Some(derived_pattern);
            }
            if let Some(expected) = sibling_plan.as_ref() {
                if expected.len() != suffix_records.len()
                    || !expected
                        .iter()
                        .zip(&suffix_records)
                        .all(|(&(key, score_bits), record)| {
                            key == record.key && score_bits == record.score_bits
                        })
                {
                    return Err(invalid_admission(format!(
                        "domain {domain_index} suffix decision keys/scores differ from its siblings"
                    )));
                }
            } else {
                let mut plan = resident_vec_with_metadata_budget(
                    suffix_records.len(),
                    GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
                    "resident sibling literal plan",
                    budget,
                    deadline,
                )?;
                plan.extend(
                    suffix_records
                        .iter()
                        .map(|record| (record.key, record.score_bits)),
                );
                sibling_plan = Some(plan);
            }
            let domain = &request.wave.domains[domain_index];
            let activation = domain
                .operands
                .activation
                .slice(request.wave.domain_arena.activation.as_ref(), "activation")?;
            let beta = domain
                .operands
                .beta
                .slice(request.wave.domain_arena.beta.as_ref(), "beta")?;
            let abs = domain
                .operands
                .abs
                .slice(request.wave.domain_arena.abs.as_ref(), "abs")?;
            let box_lower = domain
                .operands
                .box_lower
                .slice(request.wave.domain_arena.box_lower.as_ref(), "box lower")?;
            let box_upper = domain
                .operands
                .box_upper
                .slice(request.wave.domain_arena.box_upper.as_ref(), "box upper")?;
            let cached_la = domain
                .operands
                .cached_la
                .slice(request.wave.domain_arena.cached_la.as_ref(), "cached-lA")?;
            let families = [activation, beta, abs, box_lower, box_upper, cached_la];
            let logical_domain_identity_sha256 =
                hash_resident_logical_domain(phase, prefix, suffix, &families, deadline)?;
            let base_domain_identity_sha256 = hash_domain_identity_with_deadline(
                domain,
                &request.wave.domain_arena,
                domain_index,
                Some(deadline),
            )
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            if !logical_identities.insert(logical_domain_identity_sha256) {
                return Err(invalid_admission(format!(
                    "logical resident-domain identity is duplicated at domain {domain_index}"
                )));
            }
            schedule_hash.update(logical_domain_identity_sha256);
            schedule_hash.update(base_domain_identity_sha256);
            logical_domain_identities_sha256.push(logical_domain_identity_sha256);
            base_domain_identities_sha256.push(base_domain_identity_sha256);
            hash_u64(&mut schedule_hash, view.branch_pattern);
            hash_u64(&mut schedule_hash, suffix.len() as u64);
            for word in suffix {
                schedule_hash.update(word.to_le_bytes());
            }
        }
        // This v2 result bounds only the explicitly submitted children. A
        // compacted subset of the truth table does not prove parent cover or
        // authorize pruning; omitted patterns remain on the producer/TCB
        // frontier. Bind both the full cardinality and exact admitted pattern
        // set so a future cover authority cannot reinterpret this schedule.
        hash_u64(
            &mut schedule_hash,
            full_cardinality_for_group.unwrap_or(1) as u64,
        );
        hash_u64(&mut schedule_hash, base_group.child_cardinality as u64);
        for domain_index in base_group.first_domain..end_domain {
            deadline
                .poll(domain_index, "resident admitted branch patterns")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            hash_u64(
                &mut schedule_hash,
                request.domain_histories[domain_index].branch_pattern,
            );
        }
    }
    deadline
        .check("resident split-history schedule")
        .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
    if cursor != words.len() {
        return Err(invalid_admission(
            "resident split-history views do not exactly cover the arena",
        ));
    }
    Ok(ValidatedResidentHistory {
        logical_domain_identities_sha256,
        base_domain_identities_sha256,
        schedule_identity_sha256: schedule_hash.finalize().into(),
    })
}

fn materialize_resident_snapshots(
    request: &GpuBabBoundResidentWaveRequest,
    logical_domain_identities_sha256: &[[u8; 32]],
    mut snapshots: Vec<GpuBabBoundResidentDomainSnapshot>,
    budget: &mut ResidentHostAdmissionBudget,
    deadline: ResidentValidationDeadline,
) -> std::result::Result<Vec<GpuBabBoundResidentDomainSnapshot>, GpuBabBoundResidentAdmissionError>
{
    if logical_domain_identities_sha256.len() != request.wave.domains.len()
        || snapshots.capacity() < request.wave.domains.len()
    {
        return Err(GpuBabBoundResidentAdmissionError::Poison(invalid(
            "resident snapshot materializer lost its preallocated domain certificate",
        )));
    }
    for (group_index, group) in request.parent_groups.iter().enumerate() {
        deadline
            .poll(group_index, "resident snapshot groups")
            .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
        let prefix = split_words(
            group.prefix,
            request.split_history.words.as_ref(),
            "resident certified materialization prefix",
        )?;
        let base_group =
            request.wave.parent_groups.get(group_index).ok_or_else(|| {
                invalid("resident certified materialization group is out of range")
            })?;
        let end_domain = base_group
            .first_domain
            .checked_add(base_group.child_cardinality)
            .ok_or_else(|| invalid("resident certified materialization range overflows usize"))?;
        for domain_index in base_group.first_domain..end_domain {
            deadline
                .poll(domain_index, "resident snapshot domains")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            let suffix = split_words(
                request.domain_histories[domain_index].suffix,
                request.split_history.words.as_ref(),
                "resident certified materialization suffix",
            )?;
            let domain = &request.wave.domains[domain_index];
            let activation = domain
                .operands
                .activation
                .slice(request.wave.domain_arena.activation.as_ref(), "activation")?;
            let beta = domain
                .operands
                .beta
                .slice(request.wave.domain_arena.beta.as_ref(), "beta")?;
            let abs = domain
                .operands
                .abs
                .slice(request.wave.domain_arena.abs.as_ref(), "abs")?;
            let box_lower = domain
                .operands
                .box_lower
                .slice(request.wave.domain_arena.box_lower.as_ref(), "box lower")?;
            let box_upper = domain
                .operands
                .box_upper
                .slice(request.wave.domain_arena.box_upper.as_ref(), "box upper")?;
            let cached_la = domain
                .operands
                .cached_la
                .slice(request.wave.domain_arena.cached_la.as_ref(), "cached-lA")?;
            let full_history_words = prefix
                .len()
                .checked_add(suffix.len())
                .ok_or_else(|| invalid("logical child history overflows usize"))?;
            let mut full_history =
                resident_vec_with_capacity(full_history_words, "resident compact history words")?;
            budget
                .charge_snapshot_capacity(
                    full_history_words,
                    full_history.capacity(),
                    size_of::<u32>(),
                )
                .map_err(|()| {
                    GpuBabBoundResidentAdmissionError::Decline(
                        GpuBabBoundResidentWaveDecline::InsufficientCapacity,
                    )
                })?;
            for chunk in prefix
                .chunks(VALIDATION_POLL_STRIDE)
                .chain(suffix.chunks(VALIDATION_POLL_STRIDE))
            {
                deadline
                    .check("resident compact history copy")
                    .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
                full_history.extend_from_slice(chunk);
            }
            deadline
                .check("resident compact history copy")
                .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
            snapshots.push(GpuBabBoundResidentDomainSnapshot {
                activation: resident_copy_slice_with_charge(
                    activation,
                    "resident compact activation values",
                    budget,
                    deadline,
                )?,
                beta: resident_copy_slice_with_charge(
                    beta,
                    "resident compact beta values",
                    budget,
                    deadline,
                )?,
                abs: resident_copy_slice_with_charge(
                    abs,
                    "resident compact abs values",
                    budget,
                    deadline,
                )?,
                box_lower: resident_copy_slice_with_charge(
                    box_lower,
                    "resident compact lower-box values",
                    budget,
                    deadline,
                )?,
                box_upper: resident_copy_slice_with_charge(
                    box_upper,
                    "resident compact upper-box values",
                    budget,
                    deadline,
                )?,
                cached_la: resident_copy_slice_with_charge(
                    cached_la,
                    "resident compact cached-lA values",
                    budget,
                    deadline,
                )?,
                history: full_history,
                logical_domain_identity_sha256: logical_domain_identities_sha256[domain_index],
            });
            #[cfg(test)]
            RESIDENT_SNAPSHOT_MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
        }
    }
    deadline
        .check("resident snapshot materialization")
        .map_err(GpuBabBoundResidentAdmissionError::Invalid)?;
    Ok(snapshots)
}

fn resident_base_shape(
    mut shape: ValidatedWaveShape,
    _expected: GpuBabBoundResidentTransferReceipt,
) -> Result<ValidatedWaveShape> {
    shape.activation_operand_bytes = 0;
    shape.beta_operand_bytes = 0;
    shape.abs_operand_bytes = 0;
    shape.box_operand_bytes = 0;
    shape.cached_la_operand_bytes = 0;
    shape.domain_operand_bytes = 0;
    Ok(shape)
}

fn resident_transfer_is_zero(receipt: GpuBabBoundResidentTransferReceipt) -> bool {
    receipt.completed_resident_transfer_units == 0
        && receipt.resident_host_to_device_bytes == 0
        && receipt.resident_device_to_device_bytes == 0
}

#[cfg(test)]
fn validate_resident_transfer_receipt(
    actual: GpuBabBoundResidentTransferReceipt,
    expected: GpuBabBoundResidentTransferReceipt,
    pending: &GpuBabBoundPendingResidentWave,
    completed: bool,
    dispatched: bool,
) -> Result<()> {
    if actual.resident_control_payload_bytes != 0 {
        return Err(invalid(
            "resident v2 control payload must remain host-only with zero device bytes",
        ));
    }
    if actual.resident_transfer_units != expected.resident_transfer_units {
        return Err(invalid(
            "resident transfer receipt does not echo the admitted unit count",
        ));
    }
    if actual.resident_host_to_device_transfer_units
        != expected.resident_host_to_device_transfer_units
        || actual.resident_host_to_device_transfer_units > actual.resident_transfer_units
    {
        return Err(invalid(
            "resident transfer receipt does not echo the admitted H2D frontier",
        ));
    }
    let derived = pending.transfer_prefix(actual.completed_resident_transfer_units)?;
    if actual != derived {
        return Err(invalid(
            "resident transfer fields are not the exact canonical completed-unit prefix",
        ));
    }
    if !dispatched
        && (actual.completed_resident_transfer_units
            > actual.resident_host_to_device_transfer_units
            || actual.resident_device_to_device_bytes != 0)
    {
        return Err(invalid(
            "predispatch resident progress cannot execute D2D without a submitted compute wave",
        ));
    }
    if completed || dispatched {
        if actual.completed_resident_transfer_units != expected.resident_transfer_units
            || actual != expected
        {
            return Err(invalid(
                "completed/postdispatch resident transfer receipt is not the exact full plan",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
fn allocated_destination_prefix(
    pending: &GpuBabBoundPendingResidentWave,
    units: usize,
) -> Result<(usize, usize)> {
    pending
        .allocation_prefixes
        .get(units)
        .copied()
        .map(|prefix| (prefix.allocated_bytes, prefix.complete_slots))
        .ok_or_else(|| {
            invalid("allocated resident buffer prefix exceeds the admitted destination units")
        })
}

#[cfg(test)]
fn validate_resident_memory_receipt(
    receipt: GpuBabBoundResidentMemoryReceipt,
    transfers: GpuBabBoundResidentTransferReceipt,
    base_memory: GpuBabBoundMemoryReceipt,
    pending: &GpuBabBoundPendingResidentWave,
    completed: bool,
    cap: usize,
) -> Result<()> {
    if receipt.resident_device_before_bytes != pending.retained_before_bytes
        || receipt.reserved_destination_bytes != pending.destination_bytes
        || receipt.planned_release_bytes != pending.released_on_commit_bytes
        || receipt.resident_slots_before != pending.resident_slots_before
        || receipt.refresh_only_slots_before != pending.refresh_only_slots_before
        || receipt.destination_slots != pending.destinations.len()
        || receipt.destination_buffer_units != pending.destination_buffer_units()
        || receipt.consumed_parent_slots != pending.consumed_parent_slots
        || receipt.explicitly_released_slots != pending.explicitly_released_slots
        || receipt.explicitly_evicted_slots != pending.explicitly_evicted_slots
        || receipt.destination_padding_bytes != 0
    {
        return Err(invalid(
            "resident memory receipt does not exactly echo the admitted transition",
        ));
    }
    let (allocated_prefix_bytes, complete_slots) =
        allocated_destination_prefix(pending, receipt.allocated_destination_buffer_units)?;
    if receipt.allocated_destination_bytes != allocated_prefix_bytes
        || receipt.allocated_destination_slots != complete_slots
    {
        return Err(invalid(
            "resident allocated bytes/slots do not equal the canonical nonzero-buffer prefix",
        ));
    }
    if completed {
        if receipt.allocated_destination_slots != pending.destinations.len()
            || receipt.allocated_destination_buffer_units != pending.destination_buffer_units()
            || receipt.allocated_destination_bytes != pending.destination_bytes
            || receipt.released_provisional_destination_slots != 0
            || receipt.released_provisional_destination_buffer_units != 0
            || receipt.released_provisional_destination_bytes != 0
            || receipt.committed_release_bytes != pending.released_on_commit_bytes
            || receipt.resident_device_after_bytes != pending.retained_after_bytes
            || receipt.resident_slots_after != pending.resident_slots_after
            || receipt.refresh_only_slots_after != pending.refresh_only_slots_after
        {
            return Err(invalid(
                "completed resident memory receipt is not the exact atomic commit equation",
            ));
        }
    } else if receipt.released_provisional_destination_slots != receipt.allocated_destination_slots
        || receipt.released_provisional_destination_buffer_units
            != receipt.allocated_destination_buffer_units
        || receipt.released_provisional_destination_bytes != receipt.allocated_destination_bytes
        || receipt.committed_release_bytes != 0
        || receipt.resident_device_after_bytes != pending.retained_before_bytes
        || receipt.resident_slots_after != pending.resident_slots_before
        || receipt.refresh_only_slots_after != pending.refresh_only_slots_before
    {
        return Err(invalid(
            "failed resident memory receipt must release every provisional destination and preserve all sources",
        ));
    }
    if !resident_transfer_is_zero(transfers)
        && receipt.allocated_destination_buffer_units != pending.destination_buffer_units()
    {
        return Err(invalid(
            "resident transfers require every disjoint nonzero destination buffer to be allocated",
        ));
    }
    if receipt.resident_queued_upload_bytes != transfers.resident_host_to_device_bytes {
        return Err(invalid(
            "resident upload staging does not equal the typed resident H2D total",
        ));
    }
    let expected_peak = base_memory
        .checked_sum()?
        .checked_add(pending.retained_before_bytes)
        .and_then(|value| value.checked_add(receipt.allocated_destination_bytes))
        .and_then(|value| value.checked_add(transfers.resident_host_to_device_bytes))
        .ok_or_else(|| invalid("resident transition peak overflows usize"))?;
    if receipt.transition_peak_device_bytes != expected_peak || expected_peak > cap {
        return Err(invalid(
            "resident transition peak is not the exact bounded no-release-netting equation",
        ));
    }
    Ok(())
}

/// Allocation-free contract codes used while a raw failure/deadline receipt
/// still controls whether the reserved journal may be rolled back.  Owned
/// diagnostics are deliberately constructed only after rollback or quarantine
/// has settled resident authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuBabBoundResidentTerminalReceiptCode {
    BaseTranscript,
    BaseRequestedCounts,
    BaseCompletedCounts,
    BaseCompletionCounts,
    BaseTightenedRows,
    BaseCompletedTightenedRows,
    BaseAuthorizedBytes,
    BaseRetainedMemory,
    BaseMemoryEquation,
    BaseTransferEquation,
    BaseTransferBounds,
    BaseTransferBoundary,
    BaseMemoryTransferEquation,
    BaseDispatchEquation,
    BaseCompletedWork,
    BasePredispatchEquation,
    BasePostdispatchEquation,
    ResidentDomainCounts,
    ResidentTransferControl,
    ResidentTransferPlan,
    ResidentTransferPrefix,
    ResidentTransferDispatch,
    ResidentMemoryEcho,
    ResidentAllocationPrefix,
    ResidentFailureMemory,
    ResidentAllocationTransfer,
    ResidentUploadStaging,
    ResidentPeak,
    ResidentCompletedMemory,
    MaintenanceReceipt,
}

impl GpuBabBoundResidentTerminalReceiptCode {
    fn message(self) -> &'static str {
        match self {
            Self::BaseTranscript => "base terminal transcript echo is invalid",
            Self::BaseRequestedCounts => "base terminal requested counts are invalid",
            Self::BaseCompletedCounts => "base failure receipt claims completed output",
            Self::BaseCompletionCounts => "base completed receipt counts are invalid",
            Self::BaseTightenedRows => "base failure receipt claims tightened rows",
            Self::BaseCompletedTightenedRows => {
                "base completed receipt tightened-row count is invalid"
            }
            Self::BaseAuthorizedBytes => "base terminal device cap echo is invalid",
            Self::BaseRetainedMemory => "base retained-memory echo is invalid",
            Self::BaseMemoryEquation => "base memory peak equation is invalid",
            Self::BaseTransferEquation => "base typed transfer equation is invalid",
            Self::BaseTransferBounds => "base failure transfer exceeds its admitted bound",
            Self::BaseTransferBoundary => {
                "base failure transfer is not at a whole typed-buffer boundary"
            }
            Self::BaseMemoryTransferEquation => {
                "base memory staging/readback does not equal its transfer receipt"
            }
            Self::BaseDispatchEquation => "base dispatch/submit equation is invalid",
            Self::BaseCompletedWork => "base completed work frontier is invalid",
            Self::BasePredispatchEquation => "base predispatch failure equation is invalid",
            Self::BasePostdispatchEquation => "base postdispatch failure equation is invalid",
            Self::ResidentDomainCounts => "resident fresh/delta count echo is invalid",
            Self::ResidentTransferControl => "resident control payload is nonzero",
            Self::ResidentTransferPlan => "resident transfer-plan echo is invalid",
            Self::ResidentTransferPrefix => "resident transfer prefix is invalid",
            Self::ResidentTransferDispatch => "resident dispatch/transfer frontier is invalid",
            Self::ResidentMemoryEcho => "resident memory transition echo is invalid",
            Self::ResidentAllocationPrefix => "resident allocation prefix is invalid",
            Self::ResidentFailureMemory => "resident failure memory equation is invalid",
            Self::ResidentAllocationTransfer => {
                "resident transfer occurred before every destination buffer was allocated"
            }
            Self::ResidentUploadStaging => "resident upload staging equation is invalid",
            Self::ResidentPeak => "resident transition peak equation is invalid",
            Self::ResidentCompletedMemory => {
                "resident completed memory transition equation is invalid"
            }
            Self::MaintenanceReceipt => "maintenance zero-work receipt is invalid",
        }
    }
}

fn checked_base_memory_sum_for_terminal(memory: GpuBabBoundMemoryReceipt) -> Option<usize> {
    memory
        .retained_graph_bytes
        .checked_add(memory.retained_phase_bytes)
        .and_then(|value| value.checked_add(memory.wave_working_bytes))
        .and_then(|value| value.checked_add(memory.queued_upload_bytes))
        .and_then(|value| value.checked_add(memory.result_readback_bytes))
}

fn validate_base_failure_transfer_for_terminal(
    transfer: GpuBabBoundTransferReceipt,
    shape: ValidatedWaveShape,
) -> std::result::Result<(), GpuBabBoundResidentTerminalReceiptCode> {
    let typed_operands = transfer
        .activation_operand_bytes
        .checked_add(transfer.beta_operand_bytes)
        .and_then(|value| value.checked_add(transfer.abs_operand_bytes))
        .and_then(|value| value.checked_add(transfer.box_operand_bytes))
        .and_then(|value| value.checked_add(transfer.cached_la_operand_bytes))
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    let host_to_device_bytes = transfer
        .domain_operand_bytes
        .checked_add(transfer.inherited_endpoint_bytes)
        .and_then(|value| value.checked_add(transfer.objective_index_bytes))
        .and_then(|value| value.checked_add(transfer.subchunk_descriptor_bytes))
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    let device_to_host_bytes = transfer
        .result_endpoint_bytes
        .checked_add(transfer.result_sidecar_bytes)
        .and_then(|value| value.checked_add(transfer.domain_outcome_sidecar_bytes))
        .and_then(|value| value.checked_add(transfer.coefficient_device_to_host_bytes))
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    if transfer.domain_operand_bytes != typed_operands
        || transfer.host_to_device_bytes != host_to_device_bytes
        || transfer.device_to_host_bytes != device_to_host_bytes
        || transfer.coefficient_device_to_host_bytes != 0
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation);
    }
    let result_endpoint_bytes = shape
        .returned_rows
        .checked_mul(ENDPOINT_BYTES_PER_ROW)
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    let result_sidecar_bytes = shape
        .returned_rows
        .checked_mul(RESULT_SIDECAR_BYTES_PER_ROW)
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    let domain_outcome_sidecar_bytes = shape
        .domains
        .checked_mul(DOMAIN_OUTCOME_SIDECAR_BYTES)
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    if transfer.activation_operand_bytes > shape.activation_operand_bytes
        || transfer.beta_operand_bytes > shape.beta_operand_bytes
        || transfer.abs_operand_bytes > shape.abs_operand_bytes
        || transfer.box_operand_bytes > shape.box_operand_bytes
        || transfer.cached_la_operand_bytes > shape.cached_la_operand_bytes
        || transfer.domain_operand_bytes > shape.domain_operand_bytes
        || transfer.inherited_endpoint_bytes > shape.inherited_endpoint_bytes
        || transfer.objective_index_bytes > shape.objective_index_bytes
        || transfer.subchunk_descriptor_bytes > shape.subchunk_descriptor_bytes
        || transfer.result_endpoint_bytes > result_endpoint_bytes
        || transfer.result_sidecar_bytes > result_sidecar_bytes
        || transfer.domain_outcome_sidecar_bytes > domain_outcome_sidecar_bytes
        || transfer.readbacks > 1
        || transfer.synchronizations > 1
        || transfer.readbacks != usize::from(transfer.device_to_host_bytes > 0)
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseTransferBounds);
    }
    for (actual, full) in [
        (
            transfer.activation_operand_bytes,
            shape.activation_operand_bytes,
        ),
        (transfer.beta_operand_bytes, shape.beta_operand_bytes),
        (transfer.abs_operand_bytes, shape.abs_operand_bytes),
        (transfer.box_operand_bytes, shape.box_operand_bytes),
        (
            transfer.cached_la_operand_bytes,
            shape.cached_la_operand_bytes,
        ),
        (
            transfer.inherited_endpoint_bytes,
            shape.inherited_endpoint_bytes,
        ),
        (transfer.objective_index_bytes, shape.objective_index_bytes),
        (
            transfer.subchunk_descriptor_bytes,
            shape.subchunk_descriptor_bytes,
        ),
        (transfer.result_endpoint_bytes, result_endpoint_bytes),
        (transfer.result_sidecar_bytes, result_sidecar_bytes),
        (
            transfer.domain_outcome_sidecar_bytes,
            domain_outcome_sidecar_bytes,
        ),
    ] {
        if actual != 0 && actual != full {
            return Err(GpuBabBoundResidentTerminalReceiptCode::BaseTransferBoundary);
        }
    }
    Ok(())
}

fn validate_base_completed_transfer_for_terminal(
    transfer: GpuBabBoundTransferReceipt,
    shape: ValidatedWaveShape,
) -> std::result::Result<(), GpuBabBoundResidentTerminalReceiptCode> {
    let typed_operands = transfer
        .activation_operand_bytes
        .checked_add(transfer.beta_operand_bytes)
        .and_then(|value| value.checked_add(transfer.abs_operand_bytes))
        .and_then(|value| value.checked_add(transfer.box_operand_bytes))
        .and_then(|value| value.checked_add(transfer.cached_la_operand_bytes))
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    let host_to_device_bytes = transfer
        .domain_operand_bytes
        .checked_add(transfer.inherited_endpoint_bytes)
        .and_then(|value| value.checked_add(transfer.objective_index_bytes))
        .and_then(|value| value.checked_add(transfer.subchunk_descriptor_bytes))
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    let device_to_host_bytes = transfer
        .result_endpoint_bytes
        .checked_add(transfer.result_sidecar_bytes)
        .and_then(|value| value.checked_add(transfer.domain_outcome_sidecar_bytes))
        .and_then(|value| value.checked_add(transfer.coefficient_device_to_host_bytes))
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    let result_endpoint_bytes = shape
        .returned_rows
        .checked_mul(ENDPOINT_BYTES_PER_ROW)
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    let result_sidecar_bytes = shape
        .returned_rows
        .checked_mul(RESULT_SIDECAR_BYTES_PER_ROW)
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    let domain_outcome_sidecar_bytes = shape
        .domains
        .checked_mul(DOMAIN_OUTCOME_SIDECAR_BYTES)
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation)?;
    if transfer.domain_operand_bytes != typed_operands
        || transfer.host_to_device_bytes != host_to_device_bytes
        || transfer.device_to_host_bytes != device_to_host_bytes
        || transfer.coefficient_device_to_host_bytes != 0
        || transfer.activation_operand_bytes != shape.activation_operand_bytes
        || transfer.beta_operand_bytes != shape.beta_operand_bytes
        || transfer.abs_operand_bytes != shape.abs_operand_bytes
        || transfer.box_operand_bytes != shape.box_operand_bytes
        || transfer.cached_la_operand_bytes != shape.cached_la_operand_bytes
        || transfer.domain_operand_bytes != shape.domain_operand_bytes
        || transfer.inherited_endpoint_bytes != shape.inherited_endpoint_bytes
        || transfer.objective_index_bytes != shape.objective_index_bytes
        || transfer.subchunk_descriptor_bytes != shape.subchunk_descriptor_bytes
        || transfer.result_endpoint_bytes != result_endpoint_bytes
        || transfer.result_sidecar_bytes != result_sidecar_bytes
        || transfer.domain_outcome_sidecar_bytes != domain_outcome_sidecar_bytes
        || transfer.readbacks != 1
        || transfer.synchronizations != 1
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseTransferEquation);
    }
    Ok(())
}

fn validate_base_completed_receipt_for_terminal(
    receipt: &GpuBabBoundBackendWaveReceipt,
    request: &GpuBabBoundWaveRequest,
    shape: ValidatedWaveShape,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    policy: GpuBabBoundPhasePolicy,
) -> std::result::Result<(), GpuBabBoundResidentTerminalReceiptCode> {
    if receipt.transcript != transcript {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseTranscript);
    }
    let groups = request.parent_groups.len();
    let domains = request.domains.len();
    let objectives = request.objective_indices.len();
    let subchunks = request.subchunks.len();
    if receipt.requested_parent_groups != groups
        || receipt.requested_domains != domains
        || receipt.objective_rows != objectives
        || receipt.requested_rows != shape.rows
        || receipt.requested_subchunks != subchunks
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseRequestedCounts);
    }
    let bounded_domains = shape
        .returned_rows
        .checked_div(objectives)
        .filter(|_| shape.returned_rows.checked_rem(objectives) == Some(0))
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseCompletionCounts)?;
    let pruned_domains = domains
        .checked_sub(bounded_domains)
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseCompletionCounts)?;
    if receipt.completed_parent_groups != groups
        || receipt.completed_domains != domains
        || receipt.completed_rows != shape.rows
        || receipt.completed_subchunks != subchunks
        || receipt.bounded_domains != bounded_domains
        || receipt.pruned_domains != pruned_domains
        || receipt.returned_rows != shape.returned_rows
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseCompletionCounts);
    }
    if receipt.tightened_rows > receipt.completed_rows {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseCompletedTightenedRows);
    }
    if receipt.authorized_device_bytes != request.max_device_bytes {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseAuthorizedBytes);
    }
    if receipt.memory.retained_graph_bytes != open_memory.retained_graph_bytes
        || receipt.memory.retained_phase_bytes != open_memory.retained_phase_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseRetainedMemory);
    }
    let accounted = checked_base_memory_sum_for_terminal(receipt.memory)
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseMemoryEquation)?;
    if accounted == 0
        || receipt.memory.peak_device_bytes != accounted
        || accounted > request.max_device_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseMemoryEquation);
    }
    validate_base_completed_transfer_for_terminal(receipt.transfers, shape)?;
    if receipt.memory.queued_upload_bytes != receipt.transfers.host_to_device_bytes
        || receipt.memory.result_readback_bytes != receipt.transfers.device_to_host_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseMemoryTransferEquation);
    }
    if receipt.waves != 1
        || receipt.dispatches != shape.required_dispatches
        || receipt.dispatches > policy.maximum_dispatches_per_wave
        || receipt.submits == 0
        || receipt.submits > receipt.dispatches
        || receipt.submits > policy.maximum_submits_per_wave
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseDispatchEquation);
    }
    if receipt.memory.wave_working_bytes == 0
        || receipt.transfers.host_to_device_bytes == 0
        || receipt.transfers.device_to_host_bytes == 0
        || receipt.dispatches == 0
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseCompletedWork);
    }
    Ok(())
}

fn validate_base_failure_receipt_for_terminal(
    receipt: &GpuBabBoundBackendWaveReceipt,
    request: &GpuBabBoundWaveRequest,
    shape: ValidatedWaveShape,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    policy: GpuBabBoundPhasePolicy,
) -> std::result::Result<(), GpuBabBoundResidentTerminalReceiptCode> {
    if receipt.transcript != transcript {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseTranscript);
    }
    if receipt.requested_parent_groups != request.parent_groups.len()
        || receipt.requested_domains != request.domains.len()
        || receipt.objective_rows != request.objective_indices.len()
        || receipt.requested_rows != shape.rows
        || receipt.requested_subchunks != request.subchunks.len()
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseRequestedCounts);
    }
    if receipt.completed_parent_groups != 0
        || receipt.completed_domains != 0
        || receipt.completed_rows != 0
        || receipt.completed_subchunks != 0
        || receipt.bounded_domains != 0
        || receipt.pruned_domains != 0
        || receipt.returned_rows != 0
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseCompletedCounts);
    }
    if receipt.tightened_rows != 0 {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseTightenedRows);
    }
    if receipt.authorized_device_bytes != request.max_device_bytes {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseAuthorizedBytes);
    }
    if receipt.memory.retained_graph_bytes != open_memory.retained_graph_bytes
        || receipt.memory.retained_phase_bytes != open_memory.retained_phase_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseRetainedMemory);
    }
    let accounted = checked_base_memory_sum_for_terminal(receipt.memory)
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::BaseMemoryEquation)?;
    if accounted == 0
        || receipt.memory.peak_device_bytes != accounted
        || accounted > request.max_device_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseMemoryEquation);
    }
    validate_base_failure_transfer_for_terminal(receipt.transfers, shape)?;
    if receipt.memory.queued_upload_bytes != receipt.transfers.host_to_device_bytes
        || receipt.memory.result_readback_bytes != receipt.transfers.device_to_host_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseMemoryTransferEquation);
    }
    if receipt.waves != 1
        || receipt.dispatches > shape.required_dispatches
        || receipt.dispatches > policy.maximum_dispatches_per_wave
        || receipt.submits > receipt.dispatches
        || receipt.submits > policy.maximum_submits_per_wave
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BaseDispatchEquation);
    }
    if receipt.dispatches == 0 {
        if receipt.submits != 0
            || receipt.transfers.synchronizations != 0
            || receipt.transfers.device_to_host_bytes != 0
            || receipt.transfers.result_endpoint_bytes != 0
            || receipt.transfers.result_sidecar_bytes != 0
            || receipt.transfers.domain_outcome_sidecar_bytes != 0
            || receipt.transfers.readbacks != 0
            || receipt.memory.result_readback_bytes != 0
        {
            return Err(GpuBabBoundResidentTerminalReceiptCode::BasePredispatchEquation);
        }
    } else if receipt.submits == 0
        || receipt.transfers.synchronizations != 1
        || receipt.memory.wave_working_bytes == 0
        || receipt.transfers.activation_operand_bytes != shape.activation_operand_bytes
        || receipt.transfers.beta_operand_bytes != shape.beta_operand_bytes
        || receipt.transfers.abs_operand_bytes != shape.abs_operand_bytes
        || receipt.transfers.box_operand_bytes != shape.box_operand_bytes
        || receipt.transfers.cached_la_operand_bytes != shape.cached_la_operand_bytes
        || receipt.transfers.inherited_endpoint_bytes != shape.inherited_endpoint_bytes
        || receipt.transfers.objective_index_bytes != shape.objective_index_bytes
        || receipt.transfers.subchunk_descriptor_bytes != shape.subchunk_descriptor_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::BasePostdispatchEquation);
    }
    Ok(())
}

fn validate_resident_failure_receipt_for_terminal(
    receipt: &GpuBabBoundBackendResidentWaveReceipt,
    request: &GpuBabBoundResidentWaveRequest,
    shape: ValidatedWaveShape,
    pending: &GpuBabBoundPendingResidentWave,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    policy: GpuBabBoundPhasePolicy,
) -> std::result::Result<(), GpuBabBoundResidentTerminalReceiptCode> {
    if receipt.fresh_domains != pending.fresh_domains
        || receipt.delta_domains != pending.delta_domains
        || receipt.fresh_domains.checked_add(receipt.delta_domains)
            != Some(request.wave.domains.len())
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentDomainCounts);
    }
    let expected = pending.planned_transfers;
    let actual = receipt.resident_transfers;
    if actual.resident_control_payload_bytes != 0 {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentTransferControl);
    }
    if actual.resident_transfer_units != expected.resident_transfer_units
        || actual.resident_host_to_device_transfer_units
            != expected.resident_host_to_device_transfer_units
        || actual.resident_host_to_device_transfer_units > actual.resident_transfer_units
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentTransferPlan);
    }
    let Some(prefix) = pending
        .transfer_prefixes
        .get(actual.completed_resident_transfer_units)
        .copied()
    else {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentTransferPrefix);
    };
    if actual != prefix {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentTransferPrefix);
    }
    let dispatched = receipt.wave.dispatches != 0;
    if (!dispatched
        && (actual.completed_resident_transfer_units
            > actual.resident_host_to_device_transfer_units
            || actual.resident_device_to_device_bytes != 0))
        || (dispatched
            && (actual.completed_resident_transfer_units != expected.resident_transfer_units
                || actual != expected))
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentTransferDispatch);
    }
    let mut resident_shape = shape;
    resident_shape.activation_operand_bytes = 0;
    resident_shape.beta_operand_bytes = 0;
    resident_shape.abs_operand_bytes = 0;
    resident_shape.box_operand_bytes = 0;
    resident_shape.cached_la_operand_bytes = 0;
    resident_shape.domain_operand_bytes = 0;
    validate_base_failure_receipt_for_terminal(
        &receipt.wave,
        &request.wave,
        resident_shape,
        transcript,
        open_memory,
        policy,
    )?;
    let memory = receipt.resident_memory;
    if memory.resident_device_before_bytes != pending.retained_before_bytes
        || memory.reserved_destination_bytes != pending.destination_bytes
        || memory.planned_release_bytes != pending.released_on_commit_bytes
        || memory.resident_slots_before != pending.resident_slots_before
        || memory.refresh_only_slots_before != pending.refresh_only_slots_before
        || memory.destination_slots != pending.destinations.len()
        || memory.destination_buffer_units != pending.destination_buffer_units
        || memory.consumed_parent_slots != pending.consumed_parent_slots
        || memory.explicitly_released_slots != pending.explicitly_released_slots
        || memory.explicitly_evicted_slots != pending.explicitly_evicted_slots
        || memory.destination_padding_bytes != 0
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentMemoryEcho);
    }
    let Some(allocation_prefix) = pending
        .allocation_prefixes
        .get(memory.allocated_destination_buffer_units)
        .copied()
    else {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentAllocationPrefix);
    };
    if memory.allocated_destination_bytes != allocation_prefix.allocated_bytes
        || memory.allocated_destination_slots != allocation_prefix.complete_slots
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentAllocationPrefix);
    }
    if memory.released_provisional_destination_slots != memory.allocated_destination_slots
        || memory.released_provisional_destination_buffer_units
            != memory.allocated_destination_buffer_units
        || memory.released_provisional_destination_bytes != memory.allocated_destination_bytes
        || memory.committed_release_bytes != 0
        || memory.resident_device_after_bytes != pending.retained_before_bytes
        || memory.resident_slots_after != pending.resident_slots_before
        || memory.refresh_only_slots_after != pending.refresh_only_slots_before
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentFailureMemory);
    }
    if !resident_transfer_is_zero(actual)
        && memory.allocated_destination_buffer_units != pending.destination_buffer_units
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentAllocationTransfer);
    }
    if memory.resident_queued_upload_bytes != actual.resident_host_to_device_bytes {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentUploadStaging);
    }
    let expected_peak = checked_base_memory_sum_for_terminal(receipt.wave.memory)
        .and_then(|value| value.checked_add(pending.retained_before_bytes))
        .and_then(|value| value.checked_add(memory.allocated_destination_bytes))
        .and_then(|value| value.checked_add(actual.resident_host_to_device_bytes))
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::ResidentPeak)?;
    if memory.transition_peak_device_bytes != expected_peak
        || expected_peak > request.wave.max_device_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentPeak);
    }
    Ok(())
}

fn validate_resident_completed_receipt_for_terminal(
    receipt: &GpuBabBoundBackendResidentWaveReceipt,
    request: &GpuBabBoundResidentWaveRequest,
    shape: ValidatedWaveShape,
    pending: &GpuBabBoundPendingResidentWave,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    policy: GpuBabBoundPhasePolicy,
) -> std::result::Result<(), GpuBabBoundResidentTerminalReceiptCode> {
    if receipt.fresh_domains != pending.fresh_domains
        || receipt.delta_domains != pending.delta_domains
        || receipt.fresh_domains.checked_add(receipt.delta_domains)
            != Some(request.wave.domains.len())
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentDomainCounts);
    }
    if receipt.resident_transfers != pending.planned_transfers {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentTransferPlan);
    }
    let mut resident_shape = shape;
    resident_shape.activation_operand_bytes = 0;
    resident_shape.beta_operand_bytes = 0;
    resident_shape.abs_operand_bytes = 0;
    resident_shape.box_operand_bytes = 0;
    resident_shape.cached_la_operand_bytes = 0;
    resident_shape.domain_operand_bytes = 0;
    validate_base_completed_receipt_for_terminal(
        &receipt.wave,
        &request.wave,
        resident_shape,
        transcript,
        open_memory,
        policy,
    )?;
    let memory = receipt.resident_memory;
    if memory.resident_device_before_bytes != pending.retained_before_bytes
        || memory.reserved_destination_bytes != pending.destination_bytes
        || memory.planned_release_bytes != pending.released_on_commit_bytes
        || memory.resident_slots_before != pending.resident_slots_before
        || memory.refresh_only_slots_before != pending.refresh_only_slots_before
        || memory.destination_slots != pending.destinations.len()
        || memory.destination_buffer_units != pending.destination_buffer_units
        || memory.consumed_parent_slots != pending.consumed_parent_slots
        || memory.explicitly_released_slots != pending.explicitly_released_slots
        || memory.explicitly_evicted_slots != pending.explicitly_evicted_slots
        || memory.destination_padding_bytes != 0
        || memory.allocated_destination_slots != pending.destinations.len()
        || memory.allocated_destination_buffer_units != pending.destination_buffer_units
        || memory.allocated_destination_bytes != pending.destination_bytes
        || memory.released_provisional_destination_slots != 0
        || memory.released_provisional_destination_buffer_units != 0
        || memory.released_provisional_destination_bytes != 0
        || memory.committed_release_bytes != pending.released_on_commit_bytes
        || memory.resident_device_after_bytes != pending.retained_after_bytes
        || memory.resident_slots_after != pending.resident_slots_after
        || memory.refresh_only_slots_after != pending.refresh_only_slots_after
        || memory.resident_queued_upload_bytes
            != receipt.resident_transfers.resident_host_to_device_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentCompletedMemory);
    }
    let expected_peak = checked_base_memory_sum_for_terminal(receipt.wave.memory)
        .and_then(|value| value.checked_add(pending.retained_before_bytes))
        .and_then(|value| value.checked_add(memory.allocated_destination_bytes))
        .and_then(|value| {
            value.checked_add(receipt.resident_transfers.resident_host_to_device_bytes)
        })
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::ResidentPeak)?;
    if memory.transition_peak_device_bytes != expected_peak
        || expected_peak > request.wave.max_device_bytes
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::ResidentPeak);
    }
    Ok(())
}

fn validate_maintenance_failure_receipt_for_terminal(
    receipt: &GpuBabBoundBackendResidentMaintenanceReceipt,
    pending: &GpuBabBoundSealedPendingResidentMaintenance,
) -> std::result::Result<(), GpuBabBoundResidentTerminalReceiptCode> {
    if *receipt == pending.predispatch_failure_receipt() {
        Ok(())
    } else {
        Err(GpuBabBoundResidentTerminalReceiptCode::MaintenanceReceipt)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuBabBoundCompletedResidentValidationError {
    Deadline,
    ResourceContract(GpuBabBoundResidentTerminalReceiptCode),
    OutcomeCoverage,
    OutcomeAssociation,
    BoundedDomainCount,
    RowCount,
    CanonicalIndex,
    RowAssociation,
    RowStatus,
    RowInterval,
    RowDisjoint,
    TightenedRowCount,
}

impl GpuBabBoundCompletedResidentValidationError {
    fn message(self) -> &'static str {
        match self {
            Self::Deadline => "resident completed validation deadline expired",
            Self::ResourceContract(code) => code.message(),
            Self::OutcomeCoverage => {
                "resident completed outcomes do not exactly cover admitted destinations"
            }
            Self::OutcomeAssociation => "resident completed outcome association is not exact",
            Self::BoundedDomainCount => "resident bounded-domain count is invalid",
            Self::RowCount => "resident completed row count is invalid",
            Self::CanonicalIndex => "resident completed canonical row index is invalid",
            Self::RowAssociation => "resident completed row association is not exact",
            Self::RowStatus => "resident completed row status or taint is nonzero",
            Self::RowInterval => "resident completed row interval is not finite and ordered",
            Self::RowDisjoint => "resident completed row is disjoint from its inherited interval",
            Self::TightenedRowCount => "resident completed tightened-row count is not exact",
        }
    }
}

fn validate_completed_resident(
    domain_outcomes: Vec<GpuBabBoundBackendDomainOutcome>,
    rows: Vec<GpuBabBoundBackendRow>,
    receipt: GpuBabBoundBackendResidentWaveReceipt,
    request: &GpuBabBoundResidentWaveRequest,
    shape: ValidatedWaveShape,
    pending: &GpuBabBoundPendingResidentWave,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    policy: GpuBabBoundPhasePolicy,
    phase_deadline: Instant,
) -> std::result::Result<
    (
        Vec<GpuBabBoundBackendDomainOutcome>,
        Vec<GpuBabBoundBackendRow>,
    ),
    GpuBabBoundCompletedResidentValidationError,
> {
    validate_resident_completed_receipt_for_terminal(
        &receipt,
        request,
        shape,
        pending,
        transcript,
        open_memory,
        policy,
    )
    .map_err(GpuBabBoundCompletedResidentValidationError::ResourceContract)?;
    let deadline = ResidentValidationDeadline::new(request.wave.deadline, phase_deadline);
    if deadline.expired("resident completed outcome validation") {
        return Err(GpuBabBoundCompletedResidentValidationError::Deadline);
    }
    if domain_outcomes.len() != request.wave.domains.len()
        || pending.destinations.len() != request.wave.domains.len()
    {
        return Err(GpuBabBoundCompletedResidentValidationError::OutcomeCoverage);
    }
    let mut bounded_domains = 0usize;
    for (domain_index, (outcome, destination)) in domain_outcomes
        .iter()
        .zip(&pending.destinations)
        .enumerate()
    {
        if domain_index.is_multiple_of(VALIDATION_POLL_STRIDE)
            && deadline.expired("resident completed outcome validation")
        {
            return Err(GpuBabBoundCompletedResidentValidationError::Deadline);
        }
        let Some(domain) = request.wave.domains.get(domain_index) else {
            return Err(GpuBabBoundCompletedResidentValidationError::OutcomeCoverage);
        };
        if outcome.parent_group_id != domain.parent_group_id
            || outcome.child_ordinal != domain.child_ordinal
            || outcome.child_cardinality != domain.child_cardinality
            || outcome.domain_slot != domain.domain_slot
            || outcome.domain_identity_sha256 != destination.base_domain_identity_sha256
        {
            return Err(GpuBabBoundCompletedResidentValidationError::OutcomeAssociation);
        }
        match outcome.kind {
            GpuBabBoundBackendDomainOutcomeKind::Bounded => {
                bounded_domains = bounded_domains
                    .checked_add(1)
                    .ok_or(GpuBabBoundCompletedResidentValidationError::BoundedDomainCount)?;
            }
        }
    }
    let objective_count = request.wave.objective_indices.len();
    let expected_rows = bounded_domains
        .checked_mul(objective_count)
        .ok_or(GpuBabBoundCompletedResidentValidationError::RowCount)?;
    if rows.len() != expected_rows {
        return Err(GpuBabBoundCompletedResidentValidationError::RowCount);
    }
    let mut tightened_rows = 0usize;
    let mut row_cursor = 0usize;
    for (domain_index, (outcome, destination)) in domain_outcomes
        .iter()
        .zip(&pending.destinations)
        .enumerate()
    {
        if outcome.kind != GpuBabBoundBackendDomainOutcomeKind::Bounded {
            continue;
        }
        for objective_offset in 0..objective_count {
            if row_cursor.is_multiple_of(VALIDATION_POLL_STRIDE)
                && deadline.expired("resident completed row validation")
            {
                return Err(GpuBabBoundCompletedResidentValidationError::Deadline);
            }
            let q = domain_index
                .checked_mul(objective_count)
                .and_then(|value| value.checked_add(objective_offset))
                .ok_or(GpuBabBoundCompletedResidentValidationError::CanonicalIndex)?;
            let Some(row) = rows.get(row_cursor) else {
                return Err(GpuBabBoundCompletedResidentValidationError::RowCount);
            };
            let Some(domain) = request.wave.domains.get(domain_index) else {
                return Err(GpuBabBoundCompletedResidentValidationError::OutcomeCoverage);
            };
            let Some(&objective) = request.wave.objective_indices.get(objective_offset) else {
                return Err(GpuBabBoundCompletedResidentValidationError::CanonicalIndex);
            };
            if usize::try_from(row.q) != Ok(q)
                || row.parent_group_id != domain.parent_group_id
                || row.child_ordinal != domain.child_ordinal
                || row.child_cardinality != domain.child_cardinality
                || row.domain_slot != domain.domain_slot
                || row.domain_identity_sha256 != destination.base_domain_identity_sha256
                || row.objective_index != objective
            {
                return Err(GpuBabBoundCompletedResidentValidationError::RowAssociation);
            }
            if row.status != 0 || row.taint != 0 {
                return Err(GpuBabBoundCompletedResidentValidationError::RowStatus);
            }
            if !row.lower.is_finite() || !row.upper.is_finite() || row.lower > row.upper {
                return Err(GpuBabBoundCompletedResidentValidationError::RowInterval);
            }
            let (Some(&inherited_lower), Some(&inherited_upper)) = (
                request.wave.inherited_lower.get(q),
                request.wave.inherited_upper.get(q),
            ) else {
                return Err(GpuBabBoundCompletedResidentValidationError::CanonicalIndex);
            };
            if row.lower > inherited_upper || row.upper < inherited_lower {
                return Err(GpuBabBoundCompletedResidentValidationError::RowDisjoint);
            }
            if row.lower > inherited_lower || row.upper < inherited_upper {
                tightened_rows = tightened_rows
                    .checked_add(1)
                    .ok_or(GpuBabBoundCompletedResidentValidationError::TightenedRowCount)?;
            }
            row_cursor = row_cursor
                .checked_add(1)
                .ok_or(GpuBabBoundCompletedResidentValidationError::RowCount)?;
        }
    }
    if row_cursor != rows.len() || receipt.wave.tightened_rows != tightened_rows {
        return Err(GpuBabBoundCompletedResidentValidationError::TightenedRowCount);
    }
    if deadline.expired("resident completed publication") {
        return Err(GpuBabBoundCompletedResidentValidationError::Deadline);
    }
    Ok((domain_outcomes, rows))
}

fn core_predispatch_resident_receipt(
    request: &GpuBabBoundResidentWaveRequest,
    shape: ValidatedWaveShape,
    pending: &GpuBabBoundPendingResidentWave,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
) -> Result<GpuBabBoundBackendResidentWaveReceipt> {
    let expected_transfers = pending.expected_transfers()?;
    let resident_shape = resident_base_shape(shape, expected_transfers)?;
    let wave =
        core_predispatch_failure_receipt(&request.wave, resident_shape, transcript, open_memory);
    let mut resident_memory = pending.expected_memory(false, false);
    resident_memory.transition_peak_device_bytes = wave
        .memory
        .checked_sum()?
        .checked_add(pending.retained_before_bytes)
        .ok_or_else(|| invalid("core resident predispatch peak overflows usize"))?;
    Ok(GpuBabBoundBackendResidentWaveReceipt {
        wave,
        resident_memory,
        resident_transfers: pending.transfer_prefix(0)?,
        fresh_domains: pending.fresh_domains,
        delta_domains: pending.delta_domains,
    })
}

fn raw_resident_receipt(
    raw: &GpuBabBoundBackendResidentWaveDisposition,
) -> GpuBabBoundBackendResidentWaveReceipt {
    match raw {
        GpuBabBoundBackendResidentWaveDisposition::Completed { receipt, .. }
        | GpuBabBoundBackendResidentWaveDisposition::AcceptedFailure { receipt, .. }
        | GpuBabBoundBackendResidentWaveDisposition::DeadlineExpired { receipt, .. }
        | GpuBabBoundBackendResidentWaveDisposition::IllegalCleanDecline { receipt, .. } => {
            *receipt
        }
    }
}

fn resident_contract_failure(
    detail: String,
    receipt: GpuBabBoundBackendResidentWaveReceipt,
    host_audit: GpuBabBoundResidentHostAudit,
) -> GpuBabBoundResidentWaveDisposition {
    GpuBabBoundResidentWaveDisposition::AcceptedFailure(GpuBabBoundResidentWaveFailure {
        kind: GpuBabBoundTerminalFailureKind::ContractViolation,
        detail,
        receipt,
        receipt_validated: false,
        host_audit: Some(host_audit),
    })
}

fn resident_contract_failure_without_host_audit(
    detail: String,
    receipt: GpuBabBoundBackendResidentWaveReceipt,
) -> GpuBabBoundResidentWaveDisposition {
    GpuBabBoundResidentWaveDisposition::AcceptedFailure(GpuBabBoundResidentWaveFailure {
        kind: GpuBabBoundTerminalFailureKind::ContractViolation,
        detail,
        receipt,
        receipt_validated: false,
        host_audit: None,
    })
}

fn rollback_resident_before_raw_failure(
    lease: &mut GpuBabBoundPhaseLease<'_>,
    _request: &GpuBabBoundResidentWaveRequest,
    _shape: ValidatedWaveShape,
    pending: &GpuBabBoundSealedPendingResidentWave,
    transcript: GpuBabBoundTerminalTranscript,
    detail: &'static str,
    has_live_terminal_authority: bool,
) -> GpuBabBoundResidentWaveDisposition {
    let receipt = pending.predispatch_failure_receipt();
    let host_audit = pending.host_audit(false);
    // Acquire any proof of live issuer authority before the guarded journal
    // pass. No fallible registration lookup may occur between mutation and
    // certainty settlement.
    let mut live_guard = has_live_terminal_authority
        .then(|| {
            lease
                .registration
                .live_guard_noalloc(transcript.phase.backend)
        })
        .flatten();
    let rollback_clean = guarded_rollback_resident(&mut lease.resident_domains, pending);
    let receipt_validated = if rollback_clean {
        if let Some(guard) = live_guard.as_mut() {
            guard.poisoned = true;
            lease.poison_guarded_registry_with_known_resources();
            true
        } else {
            lease.poison_registry();
            false
        }
    } else {
        lease.resident_domains.poison_all();
        if let Some(guard) = live_guard.as_mut() {
            guard.poisoned = true;
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
        } else {
            lease.poison_registry();
        }
        false
    };
    drop(live_guard);
    GpuBabBoundResidentWaveDisposition::AcceptedFailure(GpuBabBoundResidentWaveFailure {
        kind: if receipt_validated {
            GpuBabBoundTerminalFailureKind::Backend(GpuBabBoundBackendFailureKind::AuthorityLost)
        } else {
            GpuBabBoundTerminalFailureKind::ContractViolation
        },
        detail: detail.into(),
        receipt,
        receipt_validated,
        host_audit: Some(host_audit),
    })
}

fn maintenance_endpoints_identity() -> [u8; 32] {
    Sha256::digest(b"ny.gpu-bab-bound.resident-maintenance.no-endpoints.v2\0").into()
}

fn core_resident_maintenance_receipt(
    pending: &GpuBabBoundPendingResidentMaintenance,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    completed: bool,
) -> Result<GpuBabBoundBackendResidentMaintenanceReceipt> {
    let mut memory = pending.expected_memory(completed);
    memory.transition_peak_device_bytes = open_memory
        .checked_sum()?
        .checked_add(pending.retained_before_bytes)
        .ok_or_else(|| invalid("resident maintenance peak overflows usize"))?;
    Ok(GpuBabBoundBackendResidentMaintenanceReceipt {
        transcript,
        memory,
        host_to_device_bytes: 0,
        device_to_host_bytes: 0,
        device_to_device_bytes: 0,
        control_payload_bytes: 0,
        transfer_units: 0,
        completed_transfer_units: 0,
        dispatches: 0,
        submits: 0,
        synchronizations: 0,
        readbacks: 0,
    })
}

#[cfg(test)]
fn validate_resident_maintenance_receipt(
    receipt: &GpuBabBoundBackendResidentMaintenanceReceipt,
    pending: &GpuBabBoundPendingResidentMaintenance,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    phase_device_cap: usize,
    completed: bool,
) -> Result<()> {
    if receipt.transcript != transcript
        || receipt.host_to_device_bytes != 0
        || receipt.device_to_host_bytes != 0
        || receipt.device_to_device_bytes != 0
        || receipt.control_payload_bytes != 0
        || receipt.transfer_units != 0
        || receipt.completed_transfer_units != 0
        || receipt.dispatches != 0
        || receipt.submits != 0
        || receipt.synchronizations != 0
        || receipt.readbacks != 0
    {
        return Err(invalid(
            "resident maintenance transcript/work/transfer fields are not exact zero-work echoes",
        ));
    }
    let mut expected = pending.expected_memory(completed);
    expected.transition_peak_device_bytes = open_memory
        .checked_sum()?
        .checked_add(pending.retained_before_bytes)
        .ok_or_else(|| invalid("resident maintenance peak overflows usize"))?;
    if expected.transition_peak_device_bytes > phase_device_cap || receipt.memory != expected {
        return Err(invalid(
            "resident maintenance memory receipt violates the exact no-release-netting equation",
        ));
    }
    Ok(())
}

fn validate_resident_maintenance_completed_receipt_for_terminal(
    receipt: &GpuBabBoundBackendResidentMaintenanceReceipt,
    pending: &GpuBabBoundPendingResidentMaintenance,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    phase_device_cap: usize,
) -> std::result::Result<(), GpuBabBoundResidentTerminalReceiptCode> {
    if receipt.transcript != transcript
        || receipt.host_to_device_bytes != 0
        || receipt.device_to_host_bytes != 0
        || receipt.device_to_device_bytes != 0
        || receipt.control_payload_bytes != 0
        || receipt.transfer_units != 0
        || receipt.completed_transfer_units != 0
        || receipt.dispatches != 0
        || receipt.submits != 0
        || receipt.synchronizations != 0
        || receipt.readbacks != 0
    {
        return Err(GpuBabBoundResidentTerminalReceiptCode::MaintenanceReceipt);
    }
    let peak = checked_base_memory_sum_for_terminal(open_memory)
        .and_then(|value| value.checked_add(pending.retained_before_bytes))
        .ok_or(GpuBabBoundResidentTerminalReceiptCode::MaintenanceReceipt)?;
    let mut expected = pending.expected_memory(true);
    expected.transition_peak_device_bytes = peak;
    if peak > phase_device_cap || receipt.memory != expected {
        return Err(GpuBabBoundResidentTerminalReceiptCode::MaintenanceReceipt);
    }
    Ok(())
}

fn raw_maintenance_receipt(
    raw: &GpuBabBoundBackendResidentMaintenanceDisposition,
) -> GpuBabBoundBackendResidentMaintenanceReceipt {
    match raw {
        GpuBabBoundBackendResidentMaintenanceDisposition::Completed { receipt }
        | GpuBabBoundBackendResidentMaintenanceDisposition::AcceptedFailure { receipt, .. }
        | GpuBabBoundBackendResidentMaintenanceDisposition::DeadlineExpired { receipt, .. } => {
            *receipt
        }
    }
}

fn maintenance_contract_failure(
    detail: String,
    receipt: GpuBabBoundBackendResidentMaintenanceReceipt,
    host_audit: GpuBabBoundResidentHostAudit,
) -> GpuBabBoundResidentMaintenanceDisposition {
    GpuBabBoundResidentMaintenanceDisposition::AcceptedFailure(
        GpuBabBoundResidentMaintenanceFailure {
            kind: GpuBabBoundTerminalFailureKind::ContractViolation,
            detail,
            receipt,
            receipt_validated: false,
            host_audit: Some(host_audit),
        },
    )
}

fn maintenance_contract_failure_without_host_audit(
    detail: String,
    receipt: GpuBabBoundBackendResidentMaintenanceReceipt,
) -> GpuBabBoundResidentMaintenanceDisposition {
    GpuBabBoundResidentMaintenanceDisposition::AcceptedFailure(
        GpuBabBoundResidentMaintenanceFailure {
            kind: GpuBabBoundTerminalFailureKind::ContractViolation,
            detail,
            receipt,
            receipt_validated: false,
            host_audit: None,
        },
    )
}

fn rollback_maintenance_before_raw_failure(
    lease: &mut GpuBabBoundPhaseLease<'_>,
    pending: &GpuBabBoundSealedPendingResidentMaintenance,
    transcript: GpuBabBoundTerminalTranscript,
    detail: &'static str,
    has_live_terminal_authority: bool,
) -> GpuBabBoundResidentMaintenanceDisposition {
    let receipt = pending.predispatch_failure_receipt();
    let host_audit = pending.host_audit(false);
    let mut live_guard = has_live_terminal_authority
        .then(|| {
            lease
                .registration
                .live_guard_noalloc(transcript.phase.backend)
        })
        .flatten();
    let rollback_clean = guarded_rollback_maintenance(&mut lease.resident_domains, pending);
    let receipt_validated = if rollback_clean {
        if let Some(guard) = live_guard.as_mut() {
            guard.poisoned = true;
            lease.poison_guarded_registry_with_known_resources();
            true
        } else {
            lease.poison_registry();
            false
        }
    } else {
        lease.resident_domains.poison_all();
        if let Some(guard) = live_guard.as_mut() {
            guard.poisoned = true;
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
        } else {
            lease.poison_registry();
        }
        false
    };
    drop(live_guard);
    GpuBabBoundResidentMaintenanceDisposition::AcceptedFailure(
        GpuBabBoundResidentMaintenanceFailure {
            kind: if receipt_validated {
                GpuBabBoundTerminalFailureKind::Backend(
                    GpuBabBoundBackendFailureKind::AuthorityLost,
                )
            } else {
                GpuBabBoundTerminalFailureKind::ContractViolation
            },
            detail: detail.into(),
            receipt,
            receipt_validated,
            host_audit: Some(host_audit),
        },
    )
}

fn finish_accepted_resident_maintenance(
    lease: &mut GpuBabBoundPhaseLease<'_>,
    request: &GpuBabBoundResidentMaintenanceRequest,
    pending: &mut GpuBabBoundSealedPendingResidentMaintenance,
    transcript: GpuBabBoundTerminalTranscript,
    raw: GpuBabBoundBackendResidentMaintenanceDisposition,
) -> GpuBabBoundResidentMaintenanceDisposition {
    if lease.state != LeaseState::WaveAccepted(transcript.wave_index) {
        let receipt = raw_maintenance_receipt(&raw);
        lease.resident_domains.poison_all();
        lease.state = LeaseState::Poisoned;
        lease.poison_registry();
        return maintenance_contract_failure(
            "accepted maintenance did not own the live lease state".into(),
            receipt,
            pending.host_audit(false),
        );
    }
    match lease.recheck_resident_policy_for_close() {
        GpuBabBoundResidentClosePolicyRecheck::Stable => {}
        policy_failure => {
            let receipt = raw_maintenance_receipt(&raw);
            lease.resident_domains.poison_all();
            lease.state = LeaseState::Poisoned;
            lease.poison_registry();
            let detail = if policy_failure == GpuBabBoundResidentClosePolicyRecheck::Changed {
                "maintenance policy changed before terminal publication"
            } else {
                "maintenance policy recheck panicked before terminal publication"
            };
            return maintenance_contract_failure(detail.into(), receipt, pending.host_audit(false));
        }
    }
    let registration = lease.registration;
    let mut live_guard = match registration.live_guard_noalloc(transcript.phase.backend) {
        Some(guard) => guard,
        None => {
            let receipt = raw_maintenance_receipt(&raw);
            lease.resident_domains.poison_all();
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
            return maintenance_contract_failure(
                "maintenance authority was lost during execution".into(),
                receipt,
                pending.host_audit(false),
            );
        }
    };
    if transcript.schedule_identity_sha256 != pending.schedule_identity_sha256
        || transcript.deadline != request.deadline
        || transcript.max_device_bytes != lease.phase.max_device_bytes
        || transcript.inherited_endpoints_sha256 != maintenance_endpoints_identity()
        || lease.resident_domains.installed_policy() != Some(pending.policy)
    {
        let receipt = raw_maintenance_receipt(&raw);
        lease.resident_domains.poison_all();
        live_guard.poisoned = true;
        lease.state = LeaseState::Poisoned;
        lease.issuer_claimed = false;
        drop(live_guard);
        return maintenance_contract_failure(
            "maintenance request/policy/transcript changed after accepted preflight".into(),
            receipt,
            pending.host_audit(false),
        );
    }

    match raw {
        GpuBabBoundBackendResidentMaintenanceDisposition::Completed { receipt } => {
            resident_completed_settlement_begin();
            let validation = validate_resident_maintenance_completed_receipt_for_terminal(
                &receipt,
                pending,
                transcript,
                lease.open_memory,
                lease.phase.max_device_bytes,
            );
            match validation {
                Ok(()) => {
                    let deadline =
                        ResidentValidationDeadline::new(request.deadline, lease.phase.deadline);
                    let expired_before_commit =
                        deadline.expired("maintenance precommit deadline gate");
                    let evicted_slots =
                        match guarded_commit_maintenance(&mut lease.resident_domains, pending) {
                            Ok(tokens) => tokens,
                            Err(()) => {
                                lease.resident_domains.poison_all();
                                live_guard.poisoned = true;
                                lease.state = LeaseState::Poisoned;
                                lease.issuer_claimed = false;
                                lease.resource_certainty =
                                    ResidentResourceCertainty::PoisonedUnknown;
                                resident_completed_settlement_mark();
                                let disposition = maintenance_contract_failure_without_host_audit(
                                resident_completed_terminal_detail(
                                    "maintenance commit panicked; resident state is quarantined",
                                ),
                                receipt,
                            );
                                drop(live_guard);
                                return disposition;
                            }
                        };
                    let expired_after_commit =
                        deadline.expired("maintenance postcommit deadline gate");
                    if expired_before_commit || expired_after_commit {
                        drop(evicted_slots);
                        lease.resident_domains.poison_all();
                        live_guard.poisoned = true;
                        lease.state = LeaseState::Poisoned;
                        lease.issuer_claimed = false;
                        lease.resource_certainty = ResidentResourceCertainty::PoisonedKnown;
                        resident_completed_settlement_mark();
                        let disposition =
                            GpuBabBoundResidentMaintenanceDisposition::DeadlineExpired(
                                GpuBabBoundResidentMaintenanceFailure {
                                    kind: GpuBabBoundTerminalFailureKind::Backend(
                                        GpuBabBoundBackendFailureKind::AuthorityLost,
                                    ),
                                    detail: resident_completed_terminal_detail(
                                        "maintenance completed atomically after its hard deadline; outputs were withheld",
                                    ),
                                    receipt,
                                    receipt_validated: true,
                                    host_audit: Some(pending.host_audit(true)),
                                },
                            );
                        drop(live_guard);
                        return disposition;
                    }
                    lease.state = LeaseState::Open;
                    lease.mark_resources_healthy_known();
                    resident_completed_settlement_mark();
                    let disposition = GpuBabBoundResidentMaintenanceDisposition::Completed(
                        ValidatedGpuBabBoundResidentMaintenanceResult {
                            receipt,
                            evicted_slots,
                            host_audit: pending.host_audit(true),
                        },
                    );
                    drop(live_guard);
                    disposition
                }
                Err(code) => {
                    lease.resident_domains.poison_all();
                    live_guard.poisoned = true;
                    lease.state = LeaseState::Poisoned;
                    lease.issuer_claimed = false;
                    lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                    resident_completed_settlement_mark();
                    let disposition = maintenance_contract_failure(
                        resident_completed_terminal_detail(code.message()),
                        receipt,
                        pending.host_audit(false),
                    );
                    drop(live_guard);
                    disposition
                }
            }
        }
        GpuBabBoundBackendResidentMaintenanceDisposition::AcceptedFailure {
            kind,
            detail,
            receipt,
        } => {
            let validation = validate_maintenance_failure_receipt_for_terminal(&receipt, pending);
            let rollback_known = validation.is_ok()
                && guarded_rollback_maintenance(&mut lease.resident_domains, pending);
            lease.resident_domains.poison_all();
            live_guard.poisoned = true;
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = if rollback_known {
                ResidentResourceCertainty::PoisonedKnown
            } else {
                ResidentResourceCertainty::PoisonedUnknown
            };
            let disposition = match (validation, rollback_known) {
                (Ok(()), true) => GpuBabBoundResidentMaintenanceDisposition::AcceptedFailure(
                    GpuBabBoundResidentMaintenanceFailure {
                        kind: GpuBabBoundTerminalFailureKind::Backend(kind),
                        detail,
                        receipt,
                        receipt_validated: true,
                        host_audit: Some(pending.host_audit(false)),
                    },
                ),
                (Ok(()), false) => maintenance_contract_failure(
                    "maintenance rollback panicked after a valid failure receipt".into(),
                    receipt,
                    pending.host_audit(false),
                ),
                (Err(code), _) => maintenance_contract_failure(
                    code.message().into(),
                    receipt,
                    pending.host_audit(false),
                ),
            };
            drop(live_guard);
            disposition
        }
        GpuBabBoundBackendResidentMaintenanceDisposition::DeadlineExpired { detail, receipt } => {
            let deadline = ResidentValidationDeadline::new(request.deadline, lease.phase.deadline);
            let truly_expired = deadline.expired("maintenance raw deadline terminal");
            let validation = if truly_expired {
                validate_maintenance_failure_receipt_for_terminal(&receipt, pending)
            } else {
                Err(GpuBabBoundResidentTerminalReceiptCode::MaintenanceReceipt)
            };
            let rollback_known = validation.is_ok()
                && guarded_rollback_maintenance(&mut lease.resident_domains, pending);
            lease.resident_domains.poison_all();
            live_guard.poisoned = true;
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = if rollback_known {
                ResidentResourceCertainty::PoisonedKnown
            } else {
                ResidentResourceCertainty::PoisonedUnknown
            };
            let disposition = match (truly_expired, validation, rollback_known) {
                (true, Ok(()), true) => GpuBabBoundResidentMaintenanceDisposition::DeadlineExpired(
                    GpuBabBoundResidentMaintenanceFailure {
                        kind: GpuBabBoundTerminalFailureKind::Backend(
                            GpuBabBoundBackendFailureKind::AuthorityLost,
                        ),
                        detail,
                        receipt,
                        receipt_validated: true,
                        host_audit: Some(pending.host_audit(false)),
                    },
                ),
                (true, Ok(()), false) => maintenance_contract_failure(
                    "maintenance rollback panicked after a valid deadline receipt".into(),
                    receipt,
                    pending.host_audit(false),
                ),
                (false, _, _) => maintenance_contract_failure(
                    "maintenance deadline disposition was returned early".into(),
                    receipt,
                    pending.host_audit(false),
                ),
                (true, Err(code), _) => maintenance_contract_failure(
                    code.message().into(),
                    receipt,
                    pending.host_audit(false),
                ),
            };
            drop(live_guard);
            disposition
        }
    }
}

fn finish_accepted_resident_wave(
    lease: &mut GpuBabBoundPhaseLease<'_>,
    request: &GpuBabBoundResidentWaveRequest,
    shape: ValidatedWaveShape,
    pending: &mut GpuBabBoundSealedPendingResidentWave,
    transcript: GpuBabBoundTerminalTranscript,
    raw: GpuBabBoundBackendResidentWaveDisposition,
) -> GpuBabBoundResidentWaveDisposition {
    if lease.state != LeaseState::WaveAccepted(transcript.wave_index) {
        let receipt = raw_resident_receipt(&raw);
        lease.resident_domains.poison_all();
        lease.state = LeaseState::Poisoned;
        lease.poison_registry();
        return resident_contract_failure(
            "accepted resident wave did not own the live lease state".into(),
            receipt,
            pending.host_audit(false),
        );
    }
    match lease.recheck_resident_policy_for_close() {
        GpuBabBoundResidentClosePolicyRecheck::Stable => {}
        policy_failure => {
            let receipt = raw_resident_receipt(&raw);
            lease.resident_domains.poison_all();
            lease.state = LeaseState::Poisoned;
            lease.poison_registry();
            let detail = if policy_failure == GpuBabBoundResidentClosePolicyRecheck::Changed {
                "resident policy changed before terminal publication"
            } else {
                "resident policy recheck panicked before terminal publication"
            };
            return resident_contract_failure(detail.into(), receipt, pending.host_audit(false));
        }
    }
    let registration = lease.registration;
    let mut live_guard = match registration.live_guard_noalloc(transcript.phase.backend) {
        Some(guard) => guard,
        None => {
            let receipt = raw_resident_receipt(&raw);
            lease.resident_domains.poison_all();
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
            return resident_contract_failure(
                "live registration authority was lost during resident execution".into(),
                receipt,
                pending.host_audit(false),
            );
        }
    };
    // The capability owns the exact request and pending plan moved through
    // preaccept; raw code receives immutable borrows only, and every arena is
    // immutable Arc storage. The preaccept certificate therefore remains
    // authoritative without a second workload-scaled hash/uniqueness pass.
    // Terminal publication needs only the O(1) transcript echo below.
    if transcript.schedule_identity_sha256 != pending.schedule_identity_sha256 {
        let receipt = raw_resident_receipt(&raw);
        lease.resident_domains.poison_all();
        live_guard.poisoned = true;
        lease.state = LeaseState::Poisoned;
        lease.issuer_claimed = false;
        lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
        drop(live_guard);
        return resident_contract_failure(
            "resident terminal transcript changed its immutable preaccept schedule certificate"
                .into(),
            receipt,
            pending.host_audit(false),
        );
    }
    match raw {
        GpuBabBoundBackendResidentWaveDisposition::Completed {
            domain_outcomes,
            rows,
            receipt,
        } => {
            resident_completed_settlement_begin();
            let validation = validate_completed_resident(
                domain_outcomes,
                rows,
                receipt,
                request,
                shape,
                pending,
                transcript,
                lease.open_memory,
                lease.policy,
                lease.phase.deadline,
            );
            match validation {
                Ok((domain_outcomes, rows)) => {
                    let deadline = ResidentValidationDeadline::new(
                        request.wave.deadline,
                        lease.phase.deadline,
                    );
                    let expired_before_commit =
                        deadline.expired("resident precommit deadline gate");
                    let commit = match guarded_commit_resident(&mut lease.resident_domains, pending)
                    {
                        Ok(commit) => commit,
                        Err(()) => {
                            lease.resident_domains.poison_all();
                            live_guard.poisoned = true;
                            lease.state = LeaseState::Poisoned;
                            lease.issuer_claimed = false;
                            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                            resident_completed_settlement_mark();
                            let disposition = resident_contract_failure_without_host_audit(
                                resident_completed_terminal_detail(
                                    "resident commit panicked; slot state is quarantined",
                                ),
                                receipt,
                            );
                            drop(live_guard);
                            return disposition;
                        }
                    };
                    let expired_after_commit =
                        deadline.expired("resident postcommit deadline gate");
                    if expired_before_commit || expired_after_commit {
                        drop(commit);
                        drop(domain_outcomes);
                        drop(rows);
                        lease.resident_domains.poison_all();
                        live_guard.poisoned = true;
                        lease.state = LeaseState::Poisoned;
                        lease.issuer_claimed = false;
                        lease.resource_certainty = ResidentResourceCertainty::PoisonedKnown;
                        resident_completed_settlement_mark();
                        let disposition = GpuBabBoundResidentWaveDisposition::DeadlineExpired(
                            GpuBabBoundResidentWaveFailure {
                                kind: GpuBabBoundTerminalFailureKind::Backend(
                                    GpuBabBoundBackendFailureKind::AuthorityLost,
                                ),
                                detail: resident_completed_terminal_detail(
                                    "resident wave committed atomically after its hard deadline; results and slot tokens were withheld",
                                ),
                                receipt,
                                receipt_validated: true,
                                host_audit: Some(pending.host_audit(true)),
                            },
                        );
                        drop(live_guard);
                        return disposition;
                    }
                    lease.state = LeaseState::Open;
                    lease.mark_resources_healthy_known();
                    resident_completed_settlement_mark();
                    let disposition = GpuBabBoundResidentWaveDisposition::Completed(
                        ValidatedGpuBabBoundResidentWaveResult {
                            domain_outcomes: GpuBabBoundValidatedResidentDomainOutcomes {
                                raw: domain_outcomes,
                            },
                            rows: GpuBabBoundValidatedResidentRows { raw: rows },
                            receipt: GpuBabBoundValidatedResidentWaveReceipt {
                                raw: receipt,
                                host_audit: pending.host_audit(true),
                            },
                            destination_slots: commit.destination_tokens,
                            evicted_slots: commit.evicted_tokens,
                        },
                    );
                    drop(live_guard);
                    disposition
                }
                Err(GpuBabBoundCompletedResidentValidationError::Deadline) => {
                    let commit = match guarded_commit_resident(&mut lease.resident_domains, pending)
                    {
                        Ok(commit) => commit,
                        Err(()) => {
                            lease.resident_domains.poison_all();
                            live_guard.poisoned = true;
                            lease.state = LeaseState::Poisoned;
                            lease.issuer_claimed = false;
                            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                            resident_completed_settlement_mark();
                            let disposition = resident_contract_failure_without_host_audit(
                                resident_completed_terminal_detail(
                                    "resident deadline settlement commit panicked",
                                ),
                                receipt,
                            );
                            drop(live_guard);
                            return disposition;
                        }
                    };
                    drop(commit);
                    lease.resident_domains.poison_all();
                    live_guard.poisoned = true;
                    lease.state = LeaseState::Poisoned;
                    lease.issuer_claimed = false;
                    lease.resource_certainty = ResidentResourceCertainty::PoisonedKnown;
                    resident_completed_settlement_mark();
                    let disposition = GpuBabBoundResidentWaveDisposition::DeadlineExpired(
                        GpuBabBoundResidentWaveFailure {
                            kind: GpuBabBoundTerminalFailureKind::Backend(
                                GpuBabBoundBackendFailureKind::AuthorityLost,
                            ),
                            detail: resident_completed_terminal_detail(
                                "resident result validation crossed its hard deadline; the physically completed transition was committed and outputs were withheld",
                            ),
                            receipt,
                            receipt_validated: true,
                            host_audit: Some(pending.host_audit(true)),
                        },
                    );
                    drop(live_guard);
                    disposition
                }
                Err(error) => {
                    lease.resident_domains.poison_all();
                    live_guard.poisoned = true;
                    lease.state = LeaseState::Poisoned;
                    lease.issuer_claimed = false;
                    lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                    resident_completed_settlement_mark();
                    let disposition = resident_contract_failure(
                        resident_completed_terminal_detail(error.message()),
                        receipt,
                        pending.host_audit(false),
                    );
                    drop(live_guard);
                    disposition
                }
            }
        }
        GpuBabBoundBackendResidentWaveDisposition::AcceptedFailure {
            kind,
            detail,
            receipt,
        } => {
            let validation = validate_resident_failure_receipt_for_terminal(
                &receipt,
                request,
                shape,
                pending,
                transcript,
                lease.open_memory,
                lease.policy,
            );
            let rollback_known = validation.is_ok()
                && guarded_rollback_resident(&mut lease.resident_domains, pending);
            lease.resident_domains.poison_all();
            live_guard.poisoned = true;
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = if rollback_known {
                ResidentResourceCertainty::PoisonedKnown
            } else {
                ResidentResourceCertainty::PoisonedUnknown
            };
            let disposition = match (validation, rollback_known) {
                (Ok(()), true) => GpuBabBoundResidentWaveDisposition::AcceptedFailure(
                    GpuBabBoundResidentWaveFailure {
                        kind: GpuBabBoundTerminalFailureKind::Backend(kind),
                        detail,
                        receipt,
                        receipt_validated: true,
                        host_audit: Some(pending.host_audit(false)),
                    },
                ),
                (Ok(()), false) => resident_contract_failure(
                    "resident rollback panicked after a valid failure receipt".into(),
                    receipt,
                    pending.host_audit(false),
                ),
                (Err(code), _) => resident_contract_failure(
                    code.message().into(),
                    receipt,
                    pending.host_audit(false),
                ),
            };
            drop(live_guard);
            disposition
        }
        GpuBabBoundBackendResidentWaveDisposition::DeadlineExpired { detail, receipt } => {
            let deadline =
                ResidentValidationDeadline::new(request.wave.deadline, lease.phase.deadline);
            let timely = deadline.expired("resident raw deadline terminal");
            let validation = if timely {
                validate_resident_failure_receipt_for_terminal(
                    &receipt,
                    request,
                    shape,
                    pending,
                    transcript,
                    lease.open_memory,
                    lease.policy,
                )
            } else {
                Err(GpuBabBoundResidentTerminalReceiptCode::ResidentTransferPlan)
            };
            let rollback_known = validation.is_ok()
                && guarded_rollback_resident(&mut lease.resident_domains, pending);
            lease.resident_domains.poison_all();
            live_guard.poisoned = true;
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = if rollback_known {
                ResidentResourceCertainty::PoisonedKnown
            } else {
                ResidentResourceCertainty::PoisonedUnknown
            };
            let disposition = match (timely, validation, rollback_known) {
                (true, Ok(()), true) => GpuBabBoundResidentWaveDisposition::DeadlineExpired(
                    GpuBabBoundResidentWaveFailure {
                        kind: GpuBabBoundTerminalFailureKind::Backend(
                            GpuBabBoundBackendFailureKind::AuthorityLost,
                        ),
                        detail,
                        receipt,
                        receipt_validated: true,
                        host_audit: Some(pending.host_audit(false)),
                    },
                ),
                (true, Ok(()), false) => resident_contract_failure(
                    "resident rollback panicked after a valid deadline receipt".into(),
                    receipt,
                    pending.host_audit(false),
                ),
                (false, _, _) => resident_contract_failure(
                    "resident deadline disposition was returned early".into(),
                    receipt,
                    pending.host_audit(false),
                ),
                (true, Err(code), _) => resident_contract_failure(
                    code.message().into(),
                    receipt,
                    pending.host_audit(false),
                ),
            };
            drop(live_guard);
            disposition
        }
        GpuBabBoundBackendResidentWaveDisposition::IllegalCleanDecline { reason, receipt } => {
            lease.resident_domains.poison_all();
            live_guard.poisoned = true;
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
            drop(live_guard);
            resident_contract_failure(
                format!("backend returned illegal postaccept resident decline {reason:?}"),
                receipt,
                pending.host_audit(false),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    #[derive(Clone, Copy)]
    enum MaintenancePrepareMode {
        Retry,
        Accepted,
    }

    #[derive(Clone, Copy)]
    enum ResidentExecuteMode {
        Completed,
        MalformedRow,
        AcceptedFailure,
        MalformedFailure,
    }

    #[derive(Clone, Copy)]
    enum V1ExecuteMode {
        Completed,
        AcceptedFailure,
    }

    struct MaintenanceSession {
        policy: Option<GpuBabBoundResidentDomainPolicy>,
        policy_change_at: Option<usize>,
        policy_after_change: Option<GpuBabBoundResidentDomainPolicy>,
        policy_calls: Arc<AtomicUsize>,
        phase: GpuBabBoundPhaseDescriptor,
        transcript: GpuBabBoundPhaseTranscript,
        prepare_mode: MaintenancePrepareMode,
        accept_v1_prepare: bool,
        v1_execute_mode: V1ExecuteMode,
        v1_execute_calls: Arc<AtomicUsize>,
        resident_execute_mode: ResidentExecuteMode,
        resident_prepare_calls: Arc<AtomicUsize>,
        maintenance_prepare_calls: Arc<AtomicUsize>,
        resident_execute_calls: Arc<AtomicUsize>,
        execute_calls: Arc<AtomicUsize>,
        malformed_maintenance_completed_receipt: bool,
        resident_device_bytes: usize,
        resident_slots: usize,
        refresh_only_slots: usize,
    }

    impl GpuBabBoundBackendSession for MaintenanceSession {
        fn open_accepted(
            &mut self,
            _accepted: &GpuBabBoundAcceptedOpen<'_>,
        ) -> GpuBabBoundBackendOpen {
            panic!("maintenance test lease is already open")
        }

        fn prepare_wave(
            &mut self,
            _prepared: &GpuBabBoundPreparedWave<'_>,
        ) -> GpuBabBoundBackendPrepareDisposition {
            if self.accept_v1_prepare {
                GpuBabBoundBackendPrepareDisposition::Accepted
            } else {
                GpuBabBoundBackendPrepareDisposition::CleanDecline(
                    GpuBabBoundWaveDecline::TemporarilyUnavailable,
                )
            }
        }

        fn resident_domain_policy(&self) -> Option<GpuBabBoundResidentDomainPolicy> {
            let call = self.policy_calls.fetch_add(1, AtomicOrdering::Relaxed);
            if self
                .policy_change_at
                .is_some_and(|change_at| call >= change_at)
            {
                self.policy_after_change
            } else {
                self.policy
            }
        }

        fn prepare_resident_wave(
            &mut self,
            _prepared: &GpuBabBoundPreparedResidentWave<'_>,
        ) -> GpuBabBoundBackendResidentPrepareDisposition {
            self.resident_prepare_calls
                .fetch_add(1, AtomicOrdering::Relaxed);
            GpuBabBoundBackendResidentPrepareDisposition::Accepted
        }

        fn prepare_resident_maintenance(
            &mut self,
            _prepared: &GpuBabBoundPreparedResidentMaintenance<'_>,
        ) -> GpuBabBoundBackendResidentMaintenancePrepareDisposition {
            self.maintenance_prepare_calls
                .fetch_add(1, AtomicOrdering::Relaxed);
            match self.prepare_mode {
                MaintenancePrepareMode::Retry => {
                    GpuBabBoundBackendResidentMaintenancePrepareDisposition::TemporarilyUnavailable
                }
                MaintenancePrepareMode::Accepted => {
                    GpuBabBoundBackendResidentMaintenancePrepareDisposition::Accepted
                }
            }
        }

        fn execute_accepted(
            &mut self,
            accepted: &GpuBabBoundAcceptedWave<'_>,
        ) -> GpuBabBoundBackendWaveDisposition {
            self.v1_execute_calls.fetch_add(1, AtomicOrdering::Relaxed);
            match self.v1_execute_mode {
                V1ExecuteMode::Completed => {
                    completed_v1_disposition_for_test(accepted, &self.phase)
                }
                V1ExecuteMode::AcceptedFailure => {
                    failed_v1_disposition_for_test(accepted, &self.phase)
                }
            }
        }

        fn execute_accepted_resident(
            &mut self,
            accepted: &GpuBabBoundAcceptedResidentWave<'_>,
        ) -> GpuBabBoundBackendResidentWaveDisposition {
            self.resident_execute_calls
                .fetch_add(1, AtomicOrdering::Relaxed);
            let mut disposition = match self.resident_execute_mode {
                ResidentExecuteMode::Completed | ResidentExecuteMode::MalformedRow => {
                    let memory = accepted.planned_memory();
                    self.resident_device_bytes = memory.resident_device_after_bytes;
                    self.resident_slots = memory.resident_slots_after;
                    self.refresh_only_slots = memory.refresh_only_slots_after;
                    completed_resident_disposition_for_test(accepted, &self.phase)
                }
                ResidentExecuteMode::AcceptedFailure | ResidentExecuteMode::MalformedFailure => {
                    failed_resident_disposition_for_test(
                        accepted,
                        &self.phase,
                        matches!(
                            self.resident_execute_mode,
                            ResidentExecuteMode::MalformedFailure
                        ),
                    )
                }
            };
            if matches!(
                self.resident_execute_mode,
                ResidentExecuteMode::MalformedRow
            ) {
                if let GpuBabBoundBackendResidentWaveDisposition::Completed { rows, .. } =
                    &mut disposition
                {
                    rows[0].q = rows[0].q.checked_add(1).unwrap();
                }
            }
            disposition
        }

        fn execute_accepted_resident_maintenance(
            &mut self,
            accepted: &GpuBabBoundAcceptedResidentMaintenance<'_>,
        ) -> GpuBabBoundBackendResidentMaintenanceDisposition {
            self.execute_calls.fetch_add(1, AtomicOrdering::Relaxed);
            let memory = accepted.planned_memory();
            self.resident_device_bytes = memory.resident_device_after_bytes;
            self.resident_slots = memory.resident_slots_after;
            self.refresh_only_slots = memory.refresh_only_slots_after;
            let mut receipt = GpuBabBoundBackendResidentMaintenanceReceipt {
                transcript: accepted.transcript(),
                memory,
                host_to_device_bytes: 0,
                device_to_host_bytes: 0,
                device_to_device_bytes: 0,
                control_payload_bytes: 0,
                transfer_units: 0,
                completed_transfer_units: 0,
                dispatches: 0,
                submits: 0,
                synchronizations: 0,
                readbacks: 0,
            };
            if self.malformed_maintenance_completed_receipt {
                receipt.control_payload_bytes = 1;
            }
            GpuBabBoundBackendResidentMaintenanceDisposition::Completed { receipt }
        }

        fn close(&mut self) -> GpuBabBoundBackendCloseDisposition {
            GpuBabBoundBackendCloseDisposition::Closed(GpuBabBoundBackendCloseReceipt {
                transcript: self.transcript,
                released_graph_bytes: 0,
                released_phase_bytes: 0,
                released_resident_device_bytes: self.resident_device_bytes,
                released_resident_slots: self.resident_slots,
                released_refresh_only_slots: self.refresh_only_slots,
                released_resident_logical_slots: self
                    .resident_slots
                    .checked_add(self.refresh_only_slots)
                    .unwrap(),
            })
        }
    }

    fn zero_memory() -> GpuBabBoundMemoryReceipt {
        GpuBabBoundMemoryReceipt {
            retained_graph_bytes: 0,
            retained_phase_bytes: 0,
            wave_working_bytes: 0,
            queued_upload_bytes: 0,
            result_readback_bytes: 0,
            peak_device_bytes: 0,
        }
    }

    fn completed_v1_disposition_for_test(
        accepted: &GpuBabBoundAcceptedWave<'_>,
        phase: &GpuBabBoundPhaseDescriptor,
    ) -> GpuBabBoundBackendWaveDisposition {
        let request = accepted.request();
        let shape = request.validate_static(phase).unwrap();
        let host_to_device_bytes = shape
            .domain_operand_bytes
            .checked_add(shape.inherited_endpoint_bytes)
            .and_then(|bytes| bytes.checked_add(shape.objective_index_bytes))
            .and_then(|bytes| bytes.checked_add(shape.subchunk_descriptor_bytes))
            .unwrap();
        let result_endpoint_bytes = shape.rows * ENDPOINT_BYTES_PER_ROW;
        let result_sidecar_bytes = shape.rows * RESULT_SIDECAR_BYTES_PER_ROW;
        let domain_outcome_sidecar_bytes = request.domains.len() * DOMAIN_OUTCOME_SIDECAR_BYTES;
        let device_to_host_bytes =
            result_endpoint_bytes + result_sidecar_bytes + domain_outcome_sidecar_bytes;
        let memory = GpuBabBoundMemoryReceipt {
            retained_graph_bytes: 0,
            retained_phase_bytes: 0,
            wave_working_bytes: 1_024,
            queued_upload_bytes: host_to_device_bytes,
            result_readback_bytes: device_to_host_bytes,
            peak_device_bytes: 1_024 + host_to_device_bytes + device_to_host_bytes,
        };
        let receipt = GpuBabBoundBackendWaveReceipt {
            transcript: accepted.transcript(),
            requested_parent_groups: request.parent_groups.len(),
            completed_parent_groups: request.parent_groups.len(),
            requested_domains: request.domains.len(),
            completed_domains: request.domains.len(),
            bounded_domains: request.domains.len(),
            pruned_domains: 0,
            objective_rows: request.objective_indices.len(),
            requested_rows: shape.rows,
            completed_rows: shape.rows,
            returned_rows: shape.rows,
            requested_subchunks: request.subchunks.len(),
            completed_subchunks: request.subchunks.len(),
            authorized_device_bytes: request.max_device_bytes,
            memory,
            transfers: GpuBabBoundTransferReceipt {
                activation_operand_bytes: shape.activation_operand_bytes,
                beta_operand_bytes: shape.beta_operand_bytes,
                abs_operand_bytes: shape.abs_operand_bytes,
                box_operand_bytes: shape.box_operand_bytes,
                cached_la_operand_bytes: shape.cached_la_operand_bytes,
                domain_operand_bytes: shape.domain_operand_bytes,
                inherited_endpoint_bytes: shape.inherited_endpoint_bytes,
                objective_index_bytes: shape.objective_index_bytes,
                subchunk_descriptor_bytes: shape.subchunk_descriptor_bytes,
                host_to_device_bytes,
                result_endpoint_bytes,
                result_sidecar_bytes,
                domain_outcome_sidecar_bytes,
                coefficient_device_to_host_bytes: 0,
                device_to_host_bytes,
                readbacks: 1,
                synchronizations: 1,
            },
            dispatches: shape.required_dispatches,
            submits: 1,
            waves: 1,
            tightened_rows: 0,
        };
        let domain_outcomes = request
            .domains
            .iter()
            .enumerate()
            .map(|(index, domain)| GpuBabBoundBackendDomainOutcome {
                parent_group_id: domain.parent_group_id,
                child_ordinal: domain.child_ordinal,
                child_cardinality: domain.child_cardinality,
                domain_slot: domain.domain_slot,
                domain_identity_sha256: request.domain_identity_sha256(index).unwrap(),
                kind: GpuBabBoundBackendDomainOutcomeKind::Bounded,
            })
            .collect();
        let objective_count = request.objective_indices.len();
        let rows = (0..shape.rows)
            .map(|q| {
                let domain_index = q / objective_count;
                let objective_offset = q % objective_count;
                let domain = &request.domains[domain_index];
                GpuBabBoundBackendRow {
                    parent_group_id: domain.parent_group_id,
                    child_ordinal: domain.child_ordinal,
                    child_cardinality: domain.child_cardinality,
                    domain_slot: domain.domain_slot,
                    domain_identity_sha256: request.domain_identity_sha256(domain_index).unwrap(),
                    objective_index: request.objective_indices[objective_offset],
                    q: q as u32,
                    lower: request.inherited_lower[q],
                    upper: request.inherited_upper[q],
                    status: 0,
                    taint: 0,
                }
            })
            .collect();
        GpuBabBoundBackendWaveDisposition::Completed {
            domain_outcomes,
            rows,
            receipt,
        }
    }

    fn failed_v1_disposition_for_test(
        accepted: &GpuBabBoundAcceptedWave<'_>,
        phase: &GpuBabBoundPhaseDescriptor,
    ) -> GpuBabBoundBackendWaveDisposition {
        let request = accepted.request();
        let shape = request.validate_static(phase).unwrap();
        let mut receipt =
            core_predispatch_failure_receipt(request, shape, accepted.transcript(), zero_memory());
        receipt.memory.wave_working_bytes = 1;
        receipt.memory.peak_device_bytes = 1;
        GpuBabBoundBackendWaveDisposition::AcceptedFailure {
            kind: GpuBabBoundBackendFailureKind::Allocation,
            detail: "injected accepted v1 allocation failure".into(),
            receipt,
        }
    }

    fn completed_resident_disposition_for_test(
        accepted: &GpuBabBoundAcceptedResidentWave<'_>,
        phase: &GpuBabBoundPhaseDescriptor,
    ) -> GpuBabBoundBackendResidentWaveDisposition {
        let wave = accepted.wave();
        let shape = wave.validate_static(phase).unwrap();
        let resident_shape = resident_base_shape(shape, accepted.planned_transfers()).unwrap();
        let base_h2d = resident_shape
            .inherited_endpoint_bytes
            .checked_add(resident_shape.objective_index_bytes)
            .and_then(|value| value.checked_add(resident_shape.subchunk_descriptor_bytes))
            .unwrap();
        let endpoint_bytes = resident_shape.returned_rows * ENDPOINT_BYTES_PER_ROW;
        let sidecar_bytes = resident_shape.returned_rows * RESULT_SIDECAR_BYTES_PER_ROW;
        let outcome_bytes = resident_shape.domains * DOMAIN_OUTCOME_SIDECAR_BYTES;
        let base_d2h = endpoint_bytes + sidecar_bytes + outcome_bytes;
        let base_memory = GpuBabBoundMemoryReceipt {
            retained_graph_bytes: 0,
            retained_phase_bytes: 0,
            wave_working_bytes: 1_024,
            queued_upload_bytes: base_h2d,
            result_readback_bytes: base_d2h,
            peak_device_bytes: 1_024 + base_h2d + base_d2h,
        };
        let base_receipt = GpuBabBoundBackendWaveReceipt {
            transcript: accepted.transcript(),
            requested_parent_groups: wave.parent_groups.len(),
            completed_parent_groups: wave.parent_groups.len(),
            requested_domains: wave.domains.len(),
            completed_domains: wave.domains.len(),
            bounded_domains: wave.domains.len(),
            pruned_domains: 0,
            objective_rows: wave.objective_indices.len(),
            requested_rows: resident_shape.rows,
            completed_rows: resident_shape.rows,
            returned_rows: resident_shape.returned_rows,
            requested_subchunks: wave.subchunks.len(),
            completed_subchunks: wave.subchunks.len(),
            authorized_device_bytes: wave.max_device_bytes,
            memory: base_memory,
            transfers: GpuBabBoundTransferReceipt {
                activation_operand_bytes: 0,
                beta_operand_bytes: 0,
                abs_operand_bytes: 0,
                box_operand_bytes: 0,
                cached_la_operand_bytes: 0,
                domain_operand_bytes: 0,
                inherited_endpoint_bytes: resident_shape.inherited_endpoint_bytes,
                objective_index_bytes: resident_shape.objective_index_bytes,
                subchunk_descriptor_bytes: resident_shape.subchunk_descriptor_bytes,
                host_to_device_bytes: base_h2d,
                result_endpoint_bytes: endpoint_bytes,
                result_sidecar_bytes: sidecar_bytes,
                domain_outcome_sidecar_bytes: outcome_bytes,
                coefficient_device_to_host_bytes: 0,
                device_to_host_bytes: base_d2h,
                readbacks: 1,
                synchronizations: 1,
            },
            dispatches: resident_shape.required_dispatches,
            submits: 1,
            waves: 1,
            tightened_rows: 0,
        };
        let mut resident_memory = accepted.planned_memory();
        resident_memory.transition_peak_device_bytes = resident_memory
            .transition_peak_device_bytes
            .checked_add(base_memory.checked_sum().unwrap())
            .unwrap();
        let mut fresh_domains = 0usize;
        let mut delta_domains = 0usize;
        for (group, prepared) in wave.parent_groups.iter().zip(accepted.groups()) {
            match prepared.source_class() {
                GpuBabBoundResidentSourceClass::FreshUpload => {
                    fresh_domains += group.child_cardinality;
                }
                GpuBabBoundResidentSourceClass::RetainedDelta => {
                    delta_domains += group.child_cardinality;
                }
            }
        }
        let receipt = GpuBabBoundBackendResidentWaveReceipt {
            wave: base_receipt,
            resident_memory,
            resident_transfers: accepted.planned_transfers(),
            fresh_domains,
            delta_domains,
        };
        let domain_outcomes = wave
            .domains
            .iter()
            .zip(accepted.destinations())
            .map(|(domain, destination)| GpuBabBoundBackendDomainOutcome {
                parent_group_id: domain.parent_group_id,
                child_ordinal: domain.child_ordinal,
                child_cardinality: domain.child_cardinality,
                domain_slot: domain.domain_slot,
                domain_identity_sha256: *destination.base_domain_identity_sha256(),
                kind: GpuBabBoundBackendDomainOutcomeKind::Bounded,
            })
            .collect();
        let objective_count = wave.objective_indices.len();
        let rows = (0..resident_shape.rows)
            .map(|q| {
                let domain_index = q / objective_count;
                let objective_offset = q % objective_count;
                let domain = &wave.domains[domain_index];
                GpuBabBoundBackendRow {
                    parent_group_id: domain.parent_group_id,
                    child_ordinal: domain.child_ordinal,
                    child_cardinality: domain.child_cardinality,
                    domain_slot: domain.domain_slot,
                    domain_identity_sha256: *accepted.destinations()[domain_index]
                        .base_domain_identity_sha256(),
                    objective_index: wave.objective_indices[objective_offset],
                    q: q as u32,
                    lower: wave.inherited_lower[q],
                    upper: wave.inherited_upper[q],
                    status: 0,
                    taint: 0,
                }
            })
            .collect();
        GpuBabBoundBackendResidentWaveDisposition::Completed {
            domain_outcomes,
            rows,
            receipt,
        }
    }

    fn failed_resident_disposition_for_test(
        accepted: &GpuBabBoundAcceptedResidentWave<'_>,
        phase: &GpuBabBoundPhaseDescriptor,
        malformed: bool,
    ) -> GpuBabBoundBackendResidentWaveDisposition {
        let wave = accepted.wave();
        let shape = wave.validate_static(phase).unwrap();
        let resident_shape = resident_base_shape(shape, accepted.planned_transfers()).unwrap();
        let mut base = core_predispatch_failure_receipt(
            wave,
            resident_shape,
            accepted.transcript(),
            zero_memory(),
        );
        base.memory.wave_working_bytes = 1;
        base.memory.peak_device_bytes = 1;
        let mut memory = accepted.planned_memory();
        memory.allocated_destination_bytes = 0;
        memory.released_provisional_destination_bytes = 0;
        memory.committed_release_bytes = 0;
        memory.resident_device_after_bytes = memory.resident_device_before_bytes;
        memory.resident_queued_upload_bytes = 0;
        memory.transition_peak_device_bytes = base
            .memory
            .checked_sum()
            .unwrap()
            .checked_add(memory.resident_device_before_bytes)
            .unwrap();
        memory.allocated_destination_slots = 0;
        memory.allocated_destination_buffer_units = 0;
        memory.released_provisional_destination_slots = 0;
        memory.released_provisional_destination_buffer_units = 0;
        memory.resident_slots_after = memory.resident_slots_before;
        memory.refresh_only_slots_after = memory.refresh_only_slots_before;
        if malformed {
            memory.allocated_destination_slots = 1;
        }
        let mut fresh_domains = 0usize;
        let mut delta_domains = 0usize;
        for (group, prepared) in wave.parent_groups.iter().zip(accepted.groups()) {
            match prepared.source_class() {
                GpuBabBoundResidentSourceClass::FreshUpload => {
                    fresh_domains += group.child_cardinality;
                }
                GpuBabBoundResidentSourceClass::RetainedDelta => {
                    delta_domains += group.child_cardinality;
                }
            }
        }
        let planned_transfers = accepted.planned_transfers();
        let resident_transfers = GpuBabBoundResidentTransferReceipt {
            resident_transfer_units: planned_transfers.resident_transfer_units,
            resident_host_to_device_transfer_units: planned_transfers
                .resident_host_to_device_transfer_units,
            ..GpuBabBoundResidentTransferReceipt::default()
        };
        GpuBabBoundBackendResidentWaveDisposition::AcceptedFailure {
            kind: GpuBabBoundBackendFailureKind::Allocation,
            detail: "injected accepted resident allocation failure".into(),
            receipt: GpuBabBoundBackendResidentWaveReceipt {
                wave: base,
                resident_memory: memory,
                resident_transfers,
                fresh_domains,
                delta_domains,
            },
        }
    }

    fn plan_wave_for_test(
        state: &GpuBabBoundResidentDomainState,
        request: &GpuBabBoundResidentWaveRequest,
        session_nonce_sha256: [u8; 32],
    ) -> std::result::Result<GpuBabBoundPendingResidentWave, GpuBabBoundResidentAdmissionError>
    {
        let phase = phase();
        let policy = state.installed_policy().unwrap();
        let candidate = resident_candidate_size(request, None).unwrap();
        let mut budget = ResidentHostAdmissionBudget::new(
            policy.maximum_retained_v2_core_host_charged_bytes,
            state.core_host_charged_bytes().unwrap(),
            candidate.host_metadata_charge_bytes,
            candidate.logical_payload_bytes,
        )
        .unwrap();
        let shape = request
            .wave
            .validate_static_with_resident_budget(&phase, Some(&mut budget), None)
            .unwrap();
        state.plan_wave(
            request,
            &phase,
            shape,
            candidate,
            session_nonce_sha256,
            zero_memory(),
            &mut budget,
            ResidentValidationDeadline::new(request.wave.deadline, phase.deadline),
        )
    }

    fn validate_history_for_test(
        request: &GpuBabBoundResidentWaveRequest,
        _materialize_snapshots: bool,
    ) -> std::result::Result<ValidatedResidentHistory, GpuBabBoundResidentAdmissionError> {
        let phase = phase();
        let shape = request.wave.validate_static(&phase).unwrap();
        let mut budget = ResidentHostAdmissionBudget::new(usize::MAX, 0, 0, 0).unwrap();
        validate_resident_history_structure(
            request,
            &phase,
            shape.schedule_identity_sha256,
            &mut budget,
            ResidentValidationDeadline::new(request.wave.deadline, phase.deadline),
        )
    }

    fn f32_tensor(
        role: GpuBabBoundF32TensorRole,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> GpuBabBoundF32Tensor {
        GpuBabBoundF32Tensor {
            role,
            shape,
            values: GpuBabBoundOwnedSlice::new(values),
        }
    }

    fn phase() -> GpuBabBoundPhaseDescriptor {
        GpuBabBoundPhaseDescriptor::new(
            GpuBabBoundGraphPlan {
                topology_schema_version: 1,
                topology_bytes: GpuBabBoundOwnedSlice::new(vec![1_u8, 2, 3, 4]),
                f32_tensors: vec![
                    f32_tensor(
                        GpuBabBoundF32TensorRole::Parameters,
                        vec![4],
                        vec![0.25, -0.5, 0.75, 1.0],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::CertifiedErrors,
                        vec![2],
                        vec![0.0, 0.0],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::Relaxations,
                        vec![2],
                        vec![0.25, 0.75],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::InputLower,
                        vec![2],
                        vec![-1.0, -1.0],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::InputUpper,
                        vec![2],
                        vec![1.0, 1.0],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::RootLower,
                        vec![2],
                        vec![-2.0, -2.0],
                    ),
                    f32_tensor(GpuBabBoundF32TensorRole::RootUpper, vec![2], vec![2.0, 2.0]),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::ObjectiveCoefficients,
                        vec![8, 2],
                        (0..8).flat_map(|_| [1.0, -1.0]).collect(),
                    ),
                ],
                u32_tensors: vec![
                    GpuBabBoundU32Tensor {
                        role: GpuBabBoundU32TensorRole::ObjectiveIndices,
                        shape: vec![8],
                        values: GpuBabBoundOwnedSlice::new((0..8_u32).collect::<Vec<_>>()),
                    },
                    GpuBabBoundU32Tensor {
                        role: GpuBabBoundU32TensorRole::TopologyMetadata,
                        shape: vec![2],
                        values: GpuBabBoundOwnedSlice::new(vec![2_u32, 3]),
                    },
                ],
                dispatches_per_subchunk: 1,
            },
            Instant::now() + Duration::from_mins(1),
            1 << 20,
        )
        .unwrap()
    }

    fn domain(child_ordinal: usize, arena_index: usize) -> GpuBabBoundDomainTranscript {
        GpuBabBoundDomainTranscript {
            parent_group_id: 10,
            child_ordinal,
            child_cardinality: 2,
            domain_slot: 100 + child_ordinal as u64,
            operands: GpuBabBoundOperandView {
                activation: GpuBabBoundArenaRange {
                    start: arena_index * 4,
                    len: 4,
                },
                beta: GpuBabBoundArenaRange {
                    start: arena_index * 2,
                    len: 2,
                },
                abs: GpuBabBoundArenaRange {
                    start: arena_index,
                    len: 1,
                },
                box_lower: GpuBabBoundArenaRange {
                    start: arena_index,
                    len: 1,
                },
                box_upper: GpuBabBoundArenaRange {
                    start: arena_index,
                    len: 1,
                },
                cached_la: GpuBabBoundArenaRange {
                    start: arena_index * 4,
                    len: 4,
                },
            },
        }
    }

    fn base_wave() -> GpuBabBoundWaveRequest {
        GpuBabBoundWaveRequest {
            parent_groups: vec![GpuBabBoundParentGroup {
                parent_group_id: 10,
                parent_identity_sha256: [31; 32],
                first_domain: 0,
                child_cardinality: 2,
            }],
            domains: vec![domain(0, 0), domain(1, 1)],
            domain_arena: GpuBabBoundDomainArena {
                activation: GpuBabBoundOwnedSlice::new(
                    (0..8).map(|value| value as f32 / 8.0).collect(),
                ),
                beta: GpuBabBoundOwnedSlice::new(vec![0.1; 4]),
                abs: GpuBabBoundOwnedSlice::new(vec![0.25, 0.5]),
                box_lower: GpuBabBoundOwnedSlice::new(vec![-1.0, -0.5]),
                box_upper: GpuBabBoundOwnedSlice::new(vec![1.0, 0.5]),
                cached_la: GpuBabBoundOwnedSlice::new(vec![0.5; 8]),
            },
            objective_indices: vec![1, 3],
            subchunks: vec![GpuBabBoundSubchunk {
                parent_group_id: 10,
                first_domain: 0,
                domain_count: 2,
                first_q: 0,
                row_count: 4,
            }],
            inherited_lower: vec![-1.0; 4],
            inherited_upper: vec![1.0; 4],
            deadline: Instant::now() + Duration::from_secs(30),
            max_device_bytes: 1 << 20,
        }
    }

    fn literal(node: u32, neuron: u32, phase: GpuBabBoundSplitHistoryPhase) -> [u32; 4] {
        GpuBabBoundSplitHistoryLiteral {
            phase,
            topology_node_id: node,
            neuron_index: neuron,
            score: 0.5,
        }
        .encode_words()
        .unwrap()
    }

    fn history_arena_from_slice(words: &[u32]) -> GpuBabBoundSplitHistoryArena {
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(words.len())
            .expect("test split-history reserve");
        owned.extend_from_slice(words);
        GpuBabBoundSplitHistoryArena::new(owned)
    }

    fn fresh_request_with_prefix(prefix: Vec<u32>) -> GpuBabBoundResidentWaveRequest {
        let inactive = literal(7, 3, GpuBabBoundSplitHistoryPhase::Inactive);
        let active = literal(7, 3, GpuBabBoundSplitHistoryPhase::Active);
        let prefix_len = prefix.len();
        let mut words = prefix;
        words.extend(inactive);
        words.extend(active);
        GpuBabBoundResidentWaveRequest::new(
            base_wave(),
            GpuBabBoundSplitHistoryArena::new(words),
            vec![GpuBabBoundResidentParentGroup {
                parent_group_id: 10,
                prefix: GpuBabBoundArenaRange {
                    start: 0,
                    len: prefix_len,
                },
                construction: GpuBabBoundResidentConstruction::AppendReluChildren,
                source: GpuBabBoundResidentParentSource::FreshUpload { prior: None },
            }],
            vec![
                GpuBabBoundSplitHistoryView {
                    suffix: GpuBabBoundArenaRange {
                        start: prefix_len,
                        len: 4,
                    },
                    branch_pattern: 0,
                },
                GpuBabBoundSplitHistoryView {
                    suffix: GpuBabBoundArenaRange {
                        start: prefix_len + 4,
                        len: 4,
                    },
                    branch_pattern: 1,
                },
            ],
            Vec::new(),
            Vec::new(),
        )
    }

    fn single_domain_fresh_request() -> GpuBabBoundResidentWaveRequest {
        let mut request = fresh_request_with_prefix(Vec::new());
        request.wave.parent_groups[0].child_cardinality = 1;
        request.wave.domains.truncate(1);
        request.wave.domains[0].child_cardinality = 1;
        request.wave.domain_arena.activation =
            GpuBabBoundOwnedSlice::new(vec![0.0_f32, 0.125, 0.25, 0.375]);
        request.wave.domain_arena.beta = GpuBabBoundOwnedSlice::new(vec![0.1_f32, 0.1]);
        request.wave.domain_arena.abs = GpuBabBoundOwnedSlice::new(vec![0.25_f32]);
        request.wave.domain_arena.box_lower = GpuBabBoundOwnedSlice::new(vec![-1.0_f32]);
        request.wave.domain_arena.box_upper = GpuBabBoundOwnedSlice::new(vec![1.0_f32]);
        request.wave.domain_arena.cached_la = GpuBabBoundOwnedSlice::new(vec![0.5_f32; 4]);
        request.wave.subchunks[0].domain_count = 1;
        request.wave.subchunks[0].row_count = request.wave.objective_indices.len();
        request
            .wave
            .inherited_lower
            .truncate(request.wave.objective_indices.len());
        request
            .wave
            .inherited_upper
            .truncate(request.wave.objective_indices.len());
        request.domain_histories.truncate(1);
        request.split_history =
            history_arena_from_slice(&literal(7, 3, GpuBabBoundSplitHistoryPhase::Inactive));
        request
    }

    fn policy() -> GpuBabBoundResidentDomainPolicy {
        GpuBabBoundResidentDomainPolicy {
            maximum_slots: 16,
            maximum_history_words: 1 << 12,
            maximum_retained_v2_core_host_charged_bytes: 1 << 20,
            maximum_resident_device_bytes: 1 << 20,
        }
    }

    fn planned(request: &GpuBabBoundResidentWaveRequest) -> GpuBabBoundPendingResidentWave {
        let mut state = GpuBabBoundResidentDomainState::default();
        state.ensure_policy(policy(), None, None).unwrap();
        plan_wave_for_test(&state, request, [9; 32]).unwrap()
    }

    fn full_two_slot_state_for_nonce(
        session_nonce_sha256: [u8; 32],
    ) -> (
        GpuBabBoundResidentDomainState,
        Vec<GpuBabBoundResidentSlotRef>,
    ) {
        let mut state = GpuBabBoundResidentDomainState::default();
        let mut exact = policy();
        exact.maximum_slots = 2;
        state.ensure_policy(exact, None, None).unwrap();
        let request = fresh_request_with_prefix(Vec::new());
        let mut pending = plan_wave_for_test(&state, &request, session_nonce_sha256).unwrap();
        state.reserve_accepted(&pending, None).unwrap();
        let commit = state.commit_completed(&mut pending);
        assert_eq!(state.live_counts(), (2, 0));
        (state, commit.destination_tokens)
    }

    fn full_two_slot_state() -> (
        GpuBabBoundResidentDomainState,
        Vec<GpuBabBoundResidentSlotRef>,
    ) {
        full_two_slot_state_for_nonce([9; 32])
    }

    fn maintenance_lease(
        prepare_mode: MaintenancePrepareMode,
    ) -> (
        GpuBabBoundPhaseLease<'static>,
        Vec<GpuBabBoundResidentSlotRef>,
        Arc<AtomicUsize>,
    ) {
        maintenance_lease_with_receipt_mode(prepare_mode, false)
    }

    fn maintenance_lease_with_receipt_mode(
        prepare_mode: MaintenancePrepareMode,
        malformed_maintenance_completed_receipt: bool,
    ) -> (
        GpuBabBoundPhaseLease<'static>,
        Vec<GpuBabBoundResidentSlotRef>,
        Arc<AtomicUsize>,
    ) {
        let phase = phase();
        let registration = Box::leak(Box::new(
            GpuBabBoundBackendRegistration::new([77; 32]).unwrap(),
        ));
        let (identity, claim) = registration.claim(&phase);
        claim.unwrap();
        let (state, tokens) = full_two_slot_state_for_nonce(identity.session_nonce_sha256);
        let transcript = GpuBabBoundPhaseTranscript::expected(identity, &phase);
        let policy = state.installed_policy().unwrap();
        let resident_device_bytes = state.resident_bytes().unwrap();
        let (resident, refresh) = state.live_counts();
        let execute_calls = Arc::new(AtomicUsize::new(0));
        let resident_prepare_calls = Arc::new(AtomicUsize::new(0));
        let session = MaintenanceSession {
            policy: Some(policy),
            policy_change_at: None,
            policy_after_change: None,
            policy_calls: Arc::new(AtomicUsize::new(0)),
            phase: phase.clone(),
            transcript,
            prepare_mode,
            accept_v1_prepare: false,
            v1_execute_mode: V1ExecuteMode::Completed,
            v1_execute_calls: Arc::new(AtomicUsize::new(0)),
            resident_execute_mode: ResidentExecuteMode::Completed,
            resident_prepare_calls,
            maintenance_prepare_calls: Arc::new(AtomicUsize::new(0)),
            resident_execute_calls: Arc::new(AtomicUsize::new(0)),
            execute_calls: Arc::clone(&execute_calls),
            malformed_maintenance_completed_receipt,
            resident_device_bytes,
            resident_slots: resident,
            refresh_only_slots: refresh,
        };
        let lease = GpuBabBoundPhaseLease {
            phase,
            policy: GpuBabBoundPhasePolicy {
                max_device_bytes: 1 << 20,
                preferred_domains_per_wave: 2,
                minimum_domains_per_wave: 1,
                maximum_domains_per_wave: 16,
                maximum_objectives: 16,
                maximum_dispatches_per_wave: 64,
                maximum_submits_per_wave: 16,
            },
            transcript,
            open_memory: zero_memory(),
            registration,
            session: Some(Box::new(session)),
            last_wave_index: 1,
            state: LeaseState::Open,
            resource_certainty: ResidentResourceCertainty::HealthyKnown,
            issuer_claimed: true,
            abandoned_terminal: None,
            abandoned_resident_terminal: None,
            abandoned_resident_maintenance_terminal: None,
            resident_domains: state,
        };
        (lease, tokens, execute_calls)
    }

    fn large_maintenance_lease(
        slot_count: usize,
    ) -> (
        GpuBabBoundPhaseLease<'static>,
        Vec<GpuBabBoundResidentSlotRef>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        assert!(slot_count > VALIDATION_POLL_STRIDE);
        let phase = phase();
        let registration = Box::leak(Box::new(
            GpuBabBoundBackendRegistration::new([79; 32]).unwrap(),
        ));
        let (identity, claim) = registration.claim(&phase);
        claim.unwrap();
        let exact_policy = GpuBabBoundResidentDomainPolicy {
            maximum_slots: slot_count,
            maximum_history_words: 1,
            maximum_retained_v2_core_host_charged_bytes: slot_count
                .checked_mul(GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES)
                .and_then(|bytes| bytes.checked_mul(2))
                .unwrap(),
            maximum_resident_device_bytes: 1,
        };
        let mut state = GpuBabBoundResidentDomainState::default();
        state.ensure_policy(exact_policy, None, None).unwrap();
        let mut tokens = Vec::with_capacity(slot_count);
        for (slot_index, slot) in state.slots.iter_mut().enumerate() {
            let mut logical_domain_identity_sha256 = [0_u8; 32];
            logical_domain_identity_sha256[..size_of::<u64>()]
                .copy_from_slice(&(slot_index as u64).to_le_bytes());
            *slot = GpuBabBoundResidentSlotState::Live(GpuBabBoundResidentLiveSlot {
                generation: 1,
                snapshot: GpuBabBoundResidentDomainSnapshot {
                    activation: Vec::new(),
                    beta: Vec::new(),
                    abs: Vec::new(),
                    box_lower: Vec::new(),
                    box_upper: Vec::new(),
                    cached_la: Vec::new(),
                    history: Vec::new(),
                    logical_domain_identity_sha256,
                },
                layout: GpuBabBoundResidentSlotLayout::default(),
                presence: GpuBabBoundResidentPresence::Resident,
                in_flight: false,
            });
            tokens.push(GpuBabBoundResidentSlotRef {
                session_nonce_sha256: identity.session_nonce_sha256,
                logical_domain_identity_sha256,
                slot_index: u32::try_from(slot_index).unwrap(),
                generation: 1,
            });
        }
        state.completed_waves = 1;
        let transcript = GpuBabBoundPhaseTranscript::expected(identity, &phase);
        let maintenance_prepare_calls = Arc::new(AtomicUsize::new(0));
        let execute_calls = Arc::new(AtomicUsize::new(0));
        let session = MaintenanceSession {
            policy: Some(exact_policy),
            policy_change_at: None,
            policy_after_change: None,
            policy_calls: Arc::new(AtomicUsize::new(0)),
            phase: phase.clone(),
            transcript,
            prepare_mode: MaintenancePrepareMode::Accepted,
            accept_v1_prepare: false,
            v1_execute_mode: V1ExecuteMode::Completed,
            v1_execute_calls: Arc::new(AtomicUsize::new(0)),
            resident_execute_mode: ResidentExecuteMode::Completed,
            resident_prepare_calls: Arc::new(AtomicUsize::new(0)),
            maintenance_prepare_calls: Arc::clone(&maintenance_prepare_calls),
            resident_execute_calls: Arc::new(AtomicUsize::new(0)),
            execute_calls: Arc::clone(&execute_calls),
            malformed_maintenance_completed_receipt: false,
            resident_device_bytes: 0,
            resident_slots: slot_count,
            refresh_only_slots: 0,
        };
        let lease = GpuBabBoundPhaseLease {
            phase,
            policy: GpuBabBoundPhasePolicy {
                max_device_bytes: 1 << 20,
                preferred_domains_per_wave: 2,
                minimum_domains_per_wave: 1,
                maximum_domains_per_wave: 16,
                maximum_objectives: 16,
                maximum_dispatches_per_wave: 64,
                maximum_submits_per_wave: 16,
            },
            transcript,
            open_memory: zero_memory(),
            registration,
            session: Some(Box::new(session)),
            last_wave_index: 1,
            state: LeaseState::Open,
            resource_certainty: ResidentResourceCertainty::HealthyKnown,
            issuer_claimed: true,
            abandoned_terminal: None,
            abandoned_resident_terminal: None,
            abandoned_resident_maintenance_terminal: None,
            resident_domains: state,
        };
        (lease, tokens, maintenance_prepare_calls, execute_calls)
    }

    fn v1_refresh_only_lease(
        v1_execute_mode: V1ExecuteMode,
    ) -> (GpuBabBoundPhaseLease<'static>, Arc<AtomicUsize>) {
        let phase = phase();
        let registration = Box::leak(Box::new(
            GpuBabBoundBackendRegistration::new([80; 32]).unwrap(),
        ));
        let (identity, claim) = registration.claim(&phase);
        claim.unwrap();
        let (mut state, tokens) = full_two_slot_state_for_nonce(identity.session_nonce_sha256);
        drop(tokens);
        for slot in &mut state.slots {
            if let GpuBabBoundResidentSlotState::Live(live) = slot {
                live.presence = GpuBabBoundResidentPresence::RefreshOnly;
            }
        }
        assert_eq!(state.live_counts(), (0, 2));
        assert_eq!(state.resident_bytes().unwrap(), 0);
        let transcript = GpuBabBoundPhaseTranscript::expected(identity, &phase);
        let v1_execute_calls = Arc::new(AtomicUsize::new(0));
        let session = MaintenanceSession {
            policy: state.installed_policy(),
            policy_change_at: None,
            policy_after_change: None,
            policy_calls: Arc::new(AtomicUsize::new(0)),
            phase: phase.clone(),
            transcript,
            prepare_mode: MaintenancePrepareMode::Accepted,
            accept_v1_prepare: true,
            v1_execute_mode,
            v1_execute_calls: Arc::clone(&v1_execute_calls),
            resident_execute_mode: ResidentExecuteMode::Completed,
            resident_prepare_calls: Arc::new(AtomicUsize::new(0)),
            maintenance_prepare_calls: Arc::new(AtomicUsize::new(0)),
            resident_execute_calls: Arc::new(AtomicUsize::new(0)),
            execute_calls: Arc::new(AtomicUsize::new(0)),
            malformed_maintenance_completed_receipt: false,
            resident_device_bytes: 0,
            resident_slots: 0,
            refresh_only_slots: 2,
        };
        let lease = GpuBabBoundPhaseLease {
            phase,
            policy: GpuBabBoundPhasePolicy {
                max_device_bytes: 1 << 20,
                preferred_domains_per_wave: 2,
                minimum_domains_per_wave: 1,
                maximum_domains_per_wave: 16,
                maximum_objectives: 16,
                maximum_dispatches_per_wave: 64,
                maximum_submits_per_wave: 16,
            },
            transcript,
            open_memory: zero_memory(),
            registration,
            session: Some(Box::new(session)),
            last_wave_index: 1,
            state: LeaseState::Open,
            resource_certainty: ResidentResourceCertainty::HealthyKnown,
            issuer_claimed: true,
            abandoned_terminal: None,
            abandoned_resident_terminal: None,
            abandoned_resident_maintenance_terminal: None,
            resident_domains: state,
        };
        (lease, v1_execute_calls)
    }

    fn one_slot_resident_lease() -> (
        GpuBabBoundPhaseLease<'static>,
        GpuBabBoundResidentSlotRef,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let phase = phase();
        let registration = Box::leak(Box::new(
            GpuBabBoundBackendRegistration::new([81; 32]).unwrap(),
        ));
        let (identity, claim) = registration.claim(&phase);
        claim.unwrap();
        let mut exact_policy = policy();
        exact_policy.maximum_slots = 1;
        let mut state = GpuBabBoundResidentDomainState::default();
        state.ensure_policy(exact_policy, None, None).unwrap();
        let request = single_domain_fresh_request();
        let mut pending =
            plan_wave_for_test(&state, &request, identity.session_nonce_sha256).unwrap();
        state.reserve_accepted(&pending, None).unwrap();
        let token = state
            .commit_completed(&mut pending)
            .destination_tokens
            .pop()
            .unwrap();
        let transcript = GpuBabBoundPhaseTranscript::expected(identity, &phase);
        let maintenance_execute_calls = Arc::new(AtomicUsize::new(0));
        let resident_execute_calls = Arc::new(AtomicUsize::new(0));
        let session = MaintenanceSession {
            policy: Some(exact_policy),
            policy_change_at: None,
            policy_after_change: None,
            policy_calls: Arc::new(AtomicUsize::new(0)),
            phase: phase.clone(),
            transcript,
            prepare_mode: MaintenancePrepareMode::Accepted,
            accept_v1_prepare: false,
            v1_execute_mode: V1ExecuteMode::Completed,
            v1_execute_calls: Arc::new(AtomicUsize::new(0)),
            resident_execute_mode: ResidentExecuteMode::Completed,
            resident_prepare_calls: Arc::new(AtomicUsize::new(0)),
            maintenance_prepare_calls: Arc::new(AtomicUsize::new(0)),
            resident_execute_calls: Arc::clone(&resident_execute_calls),
            execute_calls: Arc::clone(&maintenance_execute_calls),
            malformed_maintenance_completed_receipt: false,
            resident_device_bytes: state.resident_bytes().unwrap(),
            resident_slots: 1,
            refresh_only_slots: 0,
        };
        let lease = GpuBabBoundPhaseLease {
            phase,
            policy: GpuBabBoundPhasePolicy {
                max_device_bytes: 1 << 20,
                preferred_domains_per_wave: 1,
                minimum_domains_per_wave: 1,
                maximum_domains_per_wave: 16,
                maximum_objectives: 16,
                maximum_dispatches_per_wave: 64,
                maximum_submits_per_wave: 16,
            },
            transcript,
            open_memory: zero_memory(),
            registration,
            session: Some(Box::new(session)),
            last_wave_index: 1,
            state: LeaseState::Open,
            resource_certainty: ResidentResourceCertainty::HealthyKnown,
            issuer_claimed: true,
            abandoned_terminal: None,
            abandoned_resident_terminal: None,
            abandoned_resident_maintenance_terminal: None,
            resident_domains: state,
        };
        (
            lease,
            token,
            maintenance_execute_calls,
            resident_execute_calls,
        )
    }

    fn resident_admission_lease_with_policy(
        phase: GpuBabBoundPhaseDescriptor,
        resident_execute_mode: ResidentExecuteMode,
        initial_policy: Option<GpuBabBoundResidentDomainPolicy>,
        policy_change_at: Option<usize>,
        policy_after_change: Option<GpuBabBoundResidentDomainPolicy>,
    ) -> (
        GpuBabBoundPhaseLease<'static>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let registration = Box::leak(Box::new(
            GpuBabBoundBackendRegistration::new([78; 32]).unwrap(),
        ));
        let (identity, claim) = registration.claim(&phase);
        claim.unwrap();
        let transcript = GpuBabBoundPhaseTranscript::expected(identity, &phase);
        let resident_prepare_calls = Arc::new(AtomicUsize::new(0));
        let resident_execute_calls = Arc::new(AtomicUsize::new(0));
        let policy_calls = Arc::new(AtomicUsize::new(0));
        let maintenance_prepare_calls = Arc::new(AtomicUsize::new(0));
        let maintenance_execute_calls = Arc::new(AtomicUsize::new(0));
        let session = MaintenanceSession {
            policy: initial_policy,
            policy_change_at,
            policy_after_change,
            policy_calls: Arc::clone(&policy_calls),
            phase: phase.clone(),
            transcript,
            prepare_mode: MaintenancePrepareMode::Accepted,
            accept_v1_prepare: false,
            v1_execute_mode: V1ExecuteMode::Completed,
            v1_execute_calls: Arc::new(AtomicUsize::new(0)),
            resident_execute_mode,
            resident_prepare_calls: Arc::clone(&resident_prepare_calls),
            maintenance_prepare_calls: Arc::clone(&maintenance_prepare_calls),
            resident_execute_calls: Arc::clone(&resident_execute_calls),
            execute_calls: Arc::clone(&maintenance_execute_calls),
            malformed_maintenance_completed_receipt: false,
            resident_device_bytes: 0,
            resident_slots: 0,
            refresh_only_slots: 0,
        };
        let lease = GpuBabBoundPhaseLease {
            phase,
            policy: GpuBabBoundPhasePolicy {
                max_device_bytes: 1 << 20,
                preferred_domains_per_wave: 2,
                minimum_domains_per_wave: 1,
                maximum_domains_per_wave: 16,
                maximum_objectives: 16,
                maximum_dispatches_per_wave: 64,
                maximum_submits_per_wave: 16,
            },
            transcript,
            open_memory: zero_memory(),
            registration,
            session: Some(Box::new(session)),
            last_wave_index: 0,
            state: LeaseState::Open,
            resource_certainty: ResidentResourceCertainty::HealthyKnown,
            issuer_claimed: true,
            abandoned_terminal: None,
            abandoned_resident_terminal: None,
            abandoned_resident_maintenance_terminal: None,
            resident_domains: GpuBabBoundResidentDomainState::default(),
        };
        (
            lease,
            resident_prepare_calls,
            resident_execute_calls,
            policy_calls,
            maintenance_prepare_calls,
            maintenance_execute_calls,
        )
    }

    fn resident_admission_lease_with_mode(
        phase: GpuBabBoundPhaseDescriptor,
        resident_execute_mode: ResidentExecuteMode,
    ) -> (
        GpuBabBoundPhaseLease<'static>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let (lease, prepare_calls, execute_calls, _, _, _) = resident_admission_lease_with_policy(
            phase,
            resident_execute_mode,
            Some(policy()),
            None,
            None,
        );
        (lease, prepare_calls, execute_calls)
    }

    fn resident_admission_lease(
        phase: GpuBabBoundPhaseDescriptor,
    ) -> (
        GpuBabBoundPhaseLease<'static>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        resident_admission_lease_with_mode(phase, ResidentExecuteMode::Completed)
    }

    fn large_activation_request() -> GpuBabBoundResidentWaveRequest {
        let mut request = fresh_request_with_prefix(Vec::new());
        let values_per_domain = VALIDATION_POLL_STRIDE + 1;
        request.wave.domain_arena.activation =
            GpuBabBoundOwnedSlice::new(vec![0.25_f32; values_per_domain * 2]);
        request.wave.domains[0].operands.activation = GpuBabBoundArenaRange {
            start: 0,
            len: values_per_domain,
        };
        request.wave.domains[1].operands.activation = GpuBabBoundArenaRange {
            start: values_per_domain,
            len: values_per_domain,
        };
        request
    }

    fn mixed_delta_plan() -> GpuBabBoundPendingResidentWave {
        let mut state = GpuBabBoundResidentDomainState::default();
        let mut roomy = policy();
        roomy.maximum_slots = 4;
        state.ensure_policy(roomy, None, None).unwrap();
        let initial = fresh_request_with_prefix(Vec::new());
        let mut first = plan_wave_for_test(&state, &initial, [9; 32]).unwrap();
        state.reserve_accepted(&first, None).unwrap();
        let mut tokens = state.commit_completed(&mut first).destination_tokens;
        let parent = tokens.remove(0);
        let parent_identity = *parent.logical_domain_identity_sha256();
        let mut wave = base_wave();
        wave.parent_groups[0].parent_identity_sha256 = parent_identity;
        let prefix = literal(7, 3, GpuBabBoundSplitHistoryPhase::Inactive);
        let inactive = literal(8, 4, GpuBabBoundSplitHistoryPhase::Inactive);
        let active = literal(8, 4, GpuBabBoundSplitHistoryPhase::Active);
        let mut words = Vec::new();
        words.extend(prefix);
        words.extend(inactive);
        words.extend(active);
        let request = GpuBabBoundResidentWaveRequest::new(
            wave,
            GpuBabBoundSplitHistoryArena::new(words),
            vec![GpuBabBoundResidentParentGroup {
                parent_group_id: 10,
                prefix: GpuBabBoundArenaRange { start: 0, len: 4 },
                construction: GpuBabBoundResidentConstruction::AppendReluChildren,
                source: GpuBabBoundResidentParentSource::RetainedDelta { parent },
            }],
            vec![
                GpuBabBoundSplitHistoryView {
                    suffix: GpuBabBoundArenaRange { start: 4, len: 4 },
                    branch_pattern: 0,
                },
                GpuBabBoundSplitHistoryView {
                    suffix: GpuBabBoundArenaRange { start: 8, len: 4 },
                    branch_pattern: 1,
                },
            ],
            Vec::new(),
            Vec::new(),
        );
        plan_wave_for_test(&state, &request, [9; 32]).unwrap()
    }

    fn maintenance_transcript(
        pending: &GpuBabBoundPendingResidentMaintenance,
        deadline: Instant,
    ) -> GpuBabBoundTerminalTranscript {
        let phase = phase();
        let issuer = GpuBabBoundBackendIssuerIdentity {
            backend_issuer_sha256: [3; 32],
            registration_epoch: 1,
            generation: 1,
            session_nonce_sha256: [9; 32],
        };
        GpuBabBoundTerminalTranscript {
            phase: GpuBabBoundPhaseTranscript::expected(issuer, &phase),
            wave_index: 2,
            schedule_identity_sha256: pending.schedule_identity_sha256,
            inherited_endpoints_sha256: maintenance_endpoints_identity(),
            deadline,
            max_device_bytes: phase.max_device_bytes,
        }
    }

    #[test]
    fn policy_has_absolute_core_byte_ceilings() {
        let mut candidate = policy();
        candidate.maximum_retained_v2_core_host_charged_bytes =
            GPU_BAB_BOUND_MAX_RETAINED_V2_CORE_HOST_CHARGED_BYTES;
        candidate.maximum_resident_device_bytes = GPU_BAB_BOUND_MAX_RESIDENT_DEVICE_BYTES;
        assert!(candidate.is_valid());
        candidate.maximum_retained_v2_core_host_charged_bytes = usize::MAX;
        assert!(!candidate.is_valid());
        candidate = policy();
        candidate.maximum_resident_device_bytes = usize::MAX;
        assert!(!candidate.is_valid());
        if GPU_BAB_BOUND_MAX_RETAINED_V2_CORE_HOST_CHARGED_BYTES < usize::MAX {
            candidate = policy();
            candidate.maximum_retained_v2_core_host_charged_bytes =
                GPU_BAB_BOUND_MAX_RETAINED_V2_CORE_HOST_CHARGED_BYTES.saturating_add(1);
            assert!(!candidate.is_valid());
        }
        if GPU_BAB_BOUND_MAX_RESIDENT_DEVICE_BYTES < usize::MAX {
            candidate = policy();
            candidate.maximum_resident_device_bytes =
                GPU_BAB_BOUND_MAX_RESIDENT_DEVICE_BYTES.saturating_add(1);
            assert!(!candidate.is_valid());
        }

        let mut undercharged_table = policy();
        undercharged_table.maximum_retained_v2_core_host_charged_bytes = undercharged_table
            .maximum_slots
            .checked_mul(GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES)
            .unwrap()
            - 1;
        assert!(!undercharged_table.is_valid());
        assert!(
            size_of::<GpuBabBoundResidentSlotState>()
                + size_of::<u64>()
                + size_of::<GpuBabBoundResidentConsumedSlot>()
                + size_of::<GpuBabBoundResidentSourceAudit>()
                + size_of::<GpuBabBoundResidentSlotRef>()
                + GPU_BAB_BOUND_HOST_MAINTENANCE_FIXED_BYTES_PER_OPERATION
                <= GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES
        );
    }

    #[test]
    fn policy_support_state_is_stable_and_fail_closed() {
        let mut unsupported = GpuBabBoundResidentDomainState::default();
        unsupported.observe_unsupported().unwrap();
        unsupported.observe_unsupported().unwrap();
        assert!(unsupported.ensure_policy(policy(), None, None).is_err());
        assert!(unsupported.poisoned);

        let mut installed = GpuBabBoundResidentDomainState::default();
        installed.ensure_policy(policy(), None, None).unwrap();
        assert!(installed.observe_unsupported().is_err());
        assert!(installed.poisoned);

        let mut changed = GpuBabBoundResidentDomainState::default();
        changed.ensure_policy(policy(), None, None).unwrap();
        let mut other = policy();
        other.maximum_slots -= 1;
        assert!(changed.ensure_policy(other, None, None).is_err());
        assert!(changed.poisoned);
    }

    #[test]
    fn exact_history_record_rejects_reserved_and_nonfinite_fields() {
        let encoded = literal(3, 5, GpuBabBoundSplitHistoryPhase::Active);
        assert_eq!(encoded[0], GPU_BAB_BOUND_SPLIT_HISTORY_TAG | 1);
        assert_eq!(decode_split_record(&encoded, "test", 0).unwrap().phase, 1);
        assert!(GpuBabBoundSplitHistoryLiteral {
            phase: GpuBabBoundSplitHistoryPhase::Inactive,
            topology_node_id: u32::MAX,
            neuron_index: 0,
            score: 0.0,
        }
        .encode_words()
        .is_err());
        let mut corrupt = encoded;
        corrupt[3] = f32::NAN.to_bits();
        assert!(decode_split_record(&corrupt, "test", 0).is_err());
        corrupt = encoded;
        corrupt[0] |= 2;
        assert!(decode_split_record(&corrupt, "test", 0).is_err());
    }

    #[test]
    fn shared_prefix_amplification_is_sized_before_materialization() {
        let prefix = literal(2, 1, GpuBabBoundSplitHistoryPhase::Inactive).to_vec();
        let request = fresh_request_with_prefix(prefix);
        let size = resident_candidate_size(&request, None).unwrap();
        assert_eq!(request.split_history.words().len(), 12);
        assert_eq!(size.history_words, 16);

        let mut state = GpuBabBoundResidentDomainState::default();
        let mut capped = policy();
        capped.maximum_history_words = 15;
        state.ensure_policy(capped, None, None).unwrap();
        RESIDENT_SNAPSHOT_MATERIALIZATIONS.store(0, Ordering::Relaxed);
        let result = plan_wave_for_test(&state, &request, [9; 32]);
        assert!(matches!(
            result,
            Err(GpuBabBoundResidentAdmissionError::Decline(
                GpuBabBoundResidentWaveDecline::InsufficientCapacity
            ))
        ));
        assert_eq!(
            RESIDENT_SNAPSHOT_MATERIALIZATIONS.load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn canonical_history_ranges_reject_gap_overlap_misalignment_and_trailing_words() {
        for mutation in 0..4 {
            let mut request = fresh_request_with_prefix(Vec::new());
            match mutation {
                0 => request.domain_histories[0].suffix.start = 4,
                1 => request.domain_histories[1].suffix.start = 0,
                2 => request.domain_histories[0].suffix.start = 1,
                3 => {
                    let mut words = request.split_history.words().to_vec();
                    words.extend(literal(99, 1, GpuBabBoundSplitHistoryPhase::Inactive));
                    request.split_history = GpuBabBoundSplitHistoryArena::new(words);
                }
                _ => unreachable!(),
            }
            RESIDENT_SNAPSHOT_MATERIALIZATIONS.store(0, Ordering::Relaxed);
            assert!(resident_candidate_size(&request, None).is_err());
            assert_eq!(
                RESIDENT_SNAPSHOT_MATERIALIZATIONS.load(Ordering::Relaxed),
                0
            );
        }
    }

    #[test]
    fn duplicate_literal_and_noncanonical_pattern_order_are_rejected() {
        let duplicate_prefix = literal(7, 3, GpuBabBoundSplitHistoryPhase::Inactive).to_vec();
        assert!(
            validate_history_for_test(&fresh_request_with_prefix(duplicate_prefix), false,)
                .is_err()
        );

        let mut unsorted = fresh_request_with_prefix(Vec::new());
        let mut words = Vec::new();
        words.extend(literal(7, 3, GpuBabBoundSplitHistoryPhase::Active));
        words.extend(literal(7, 3, GpuBabBoundSplitHistoryPhase::Inactive));
        unsorted.split_history = GpuBabBoundSplitHistoryArena::new(words);
        unsorted.domain_histories[0].branch_pattern = 1;
        unsorted.domain_histories[1].branch_pattern = 0;
        assert!(validate_history_for_test(&unsorted, false).is_err());
    }

    #[test]
    fn materialized_snapshot_has_one_owner_and_preflight_is_metadata_only() {
        let pending = planned(&fresh_request_with_prefix(Vec::new()));
        assert_eq!(pending.destinations.len(), 2);
        assert_eq!(pending.accepted_destinations.len(), 2);
        for ((destination, snapshot), proposed) in pending
            .destinations
            .iter()
            .zip(&pending.destination_snapshots)
            .zip(&pending.accepted_destinations)
        {
            assert_eq!(
                destination.logical_domain_identity_sha256,
                snapshot.logical_domain_identity_sha256,
            );
            assert_eq!(
                snapshot.logical_domain_identity_sha256,
                proposed.logical_domain_identity_sha256,
            );
        }
        assert_eq!(pending.schedule_identity_sha256.len(), 32);
    }

    #[test]
    fn allocation_and_transfer_progress_accept_every_canonical_mid_slot_cut() {
        let pending = planned(&fresh_request_with_prefix(Vec::new()));
        let total_units = pending.destination_buffer_units();
        assert!(total_units > pending.destinations.len());
        for units in 0..=total_units {
            let (bytes, complete_slots) = allocated_destination_prefix(&pending, units).unwrap();
            let mut memory = pending.expected_memory(false, false);
            memory.allocated_destination_buffer_units = units;
            memory.allocated_destination_bytes = bytes;
            memory.allocated_destination_slots = complete_slots;
            memory.released_provisional_destination_buffer_units = units;
            memory.released_provisional_destination_bytes = bytes;
            memory.released_provisional_destination_slots = complete_slots;
            memory.transition_peak_device_bytes = pending.retained_before_bytes + bytes;
            let transfers = pending.transfer_prefix(0).unwrap();
            validate_resident_memory_receipt(
                memory,
                transfers,
                zero_memory(),
                &pending,
                false,
                1 << 20,
            )
            .unwrap();
        }

        let full_allocation = pending.expected_memory(false, true);
        for units in 0..=pending
            .expected_transfers()
            .unwrap()
            .resident_transfer_units
        {
            let transfers = pending.transfer_prefix(units).unwrap();
            let mut memory = full_allocation;
            memory.resident_queued_upload_bytes = transfers.resident_host_to_device_bytes;
            memory.transition_peak_device_bytes = pending
                .retained_before_bytes
                .checked_add(pending.destination_bytes)
                .and_then(|value| value.checked_add(transfers.resident_host_to_device_bytes))
                .unwrap();
            validate_resident_memory_receipt(
                memory,
                transfers,
                zero_memory(),
                &pending,
                false,
                1 << 20,
            )
            .unwrap();
        }
    }

    #[test]
    fn mixed_delta_transfer_prefixes_finish_all_h2d_before_any_d2d() {
        let pending = mixed_delta_plan();
        let expected = pending.expected_transfers().unwrap();
        assert!(expected.resident_host_to_device_bytes > 0);
        assert!(expected.resident_device_to_device_bytes > 0);
        let h2d_units = pending.host_to_device_transfer_units();
        assert!(h2d_units > 0);
        assert!(h2d_units < expected.resident_transfer_units);
        for units in 0..=h2d_units {
            let prefix = pending.transfer_prefix(units).unwrap();
            assert_eq!(prefix.resident_device_to_device_bytes, 0);
            validate_resident_transfer_receipt(prefix, expected, &pending, false, false).unwrap();
        }
        let first_d2d = pending.transfer_prefix(h2d_units + 1).unwrap();
        assert!(first_d2d.resident_device_to_device_bytes > 0);
        assert!(
            validate_resident_transfer_receipt(first_d2d, expected, &pending, false, false)
                .is_err()
        );
        validate_resident_transfer_receipt(expected, expected, &pending, true, true).unwrap();
    }

    #[test]
    fn partial_allocation_cannot_claim_transfers_or_full_reservation_peak() {
        let pending = planned(&fresh_request_with_prefix(Vec::new()));
        let transfers = pending.transfer_prefix(1).unwrap();
        let mut memory = pending.expected_memory(false, false);
        memory.resident_queued_upload_bytes = transfers.resident_host_to_device_bytes;
        memory.transition_peak_device_bytes = transfers.resident_host_to_device_bytes;
        assert!(validate_resident_memory_receipt(
            memory,
            transfers,
            zero_memory(),
            &pending,
            false,
            1 << 20,
        )
        .is_err());

        let mut predispatch = pending.expected_memory(false, false);
        predispatch.transition_peak_device_bytes = pending.destination_bytes;
        assert!(validate_resident_memory_receipt(
            predispatch,
            pending.transfer_prefix(0).unwrap(),
            zero_memory(),
            &pending,
            false,
            1 << 20,
        )
        .is_err());
    }

    #[test]
    fn zero_byte_resident_presence_is_not_refresh_only() {
        let snapshot = GpuBabBoundResidentDomainSnapshot {
            activation: Vec::new(),
            beta: Vec::new(),
            abs: Vec::new(),
            box_lower: Vec::new(),
            box_upper: Vec::new(),
            cached_la: Vec::new(),
            history: Vec::new(),
            logical_domain_identity_sha256: [4; 32],
        };
        let state = GpuBabBoundResidentDomainState {
            policy_state: GpuBabBoundResidentPolicyState::Installed(policy()),
            slots: vec![GpuBabBoundResidentSlotState::Live(
                GpuBabBoundResidentLiveSlot {
                    generation: 1,
                    snapshot,
                    layout: GpuBabBoundResidentSlotLayout::default(),
                    presence: GpuBabBoundResidentPresence::Resident,
                    in_flight: false,
                },
            )],
            ..GpuBabBoundResidentDomainState::default()
        };
        assert_eq!(state.live_counts(), (1, 0));
        assert_eq!(state.resident_bytes().unwrap(), 0);
        let GpuBabBoundResidentSlotState::Live(live) = &state.slots[0] else {
            unreachable!()
        };
        let audit = live.source_audit(GpuBabBoundResidentSlotTranscript {
            session_nonce_sha256: [1; 32],
            logical_domain_identity_sha256: [4; 32],
            slot_index: 0,
            generation: 1,
        });
        assert_eq!(
            audit.presence(),
            GpuBabBoundResidentSourcePresence::Resident
        );
        assert_eq!(audit.resident_device_bytes(), 0);
    }

    #[test]
    fn oversized_consumed_sidecars_refuse_before_snapshot_copy() {
        let mut request = fresh_request_with_prefix(Vec::new());
        request.release.push(GpuBabBoundResidentSlotRef {
            session_nonce_sha256: [9; 32],
            logical_domain_identity_sha256: [8; 32],
            slot_index: 0,
            generation: 1,
        });
        request.release.push(GpuBabBoundResidentSlotRef {
            session_nonce_sha256: [9; 32],
            logical_domain_identity_sha256: [7; 32],
            slot_index: 1,
            generation: 1,
        });
        let mut state = GpuBabBoundResidentDomainState::default();
        let mut tiny = policy();
        tiny.maximum_slots = 2;
        state.ensure_policy(tiny, None, None).unwrap();
        state.completed_waves = 1;
        RESIDENT_SNAPSHOT_MATERIALIZATIONS.store(0, Ordering::Relaxed);
        assert!(matches!(
            plan_wave_for_test(&state, &request, [9; 32]),
            Err(GpuBabBoundResidentAdmissionError::Invalid(_))
        ));
        assert_eq!(
            RESIDENT_SNAPSHOT_MATERIALIZATIONS.load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn full_table_maintenance_releases_evicts_and_preserves_generation_high_water() {
        let (mut state, mut tokens) = full_two_slot_state();
        let token0 = tokens.remove(0);
        let token1 = tokens.remove(0);
        let deadline = Instant::now() + Duration::from_mins(1);
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(vec![token0], vec![token1], deadline);
        let materializations = RESIDENT_SNAPSHOT_MATERIALIZATIONS.load(Ordering::Relaxed);
        let mut pending = state
            .plan_maintenance(
                &request,
                &phase(),
                [9; 32],
                zero_memory(),
                ResidentValidationDeadline::new(request.deadline, request.deadline),
            )
            .unwrap();
        assert_eq!(pending.policy.maximum_slots, 2);
        assert_eq!(
            pending.host_audit.retained_v2_core_host_peak_charged_bytes,
            state.core_host_charged_bytes().unwrap()
        );
        assert_eq!(
            pending.host_audit.history_peak_words,
            state.history_words().unwrap()
        );
        state.reserve_maintenance(&pending, None).unwrap();
        let transcript = maintenance_transcript(&pending, deadline);
        let receipt =
            core_resident_maintenance_receipt(&pending, transcript, zero_memory(), true).unwrap();
        validate_resident_maintenance_receipt(
            &receipt,
            &pending,
            transcript,
            zero_memory(),
            1 << 20,
            true,
        )
        .unwrap();
        let evicted = state.commit_maintenance(&mut pending);
        assert_eq!(evicted.len(), 1);
        assert_eq!(state.live_counts(), (0, 1));
        assert_eq!(state.resident_bytes().unwrap(), 0);
        assert_eq!(
            RESIDENT_SNAPSHOT_MATERIALIZATIONS.load(Ordering::Relaxed),
            materializations
        );

        let request = GpuBabBoundResidentMaintenanceRequest::new(
            evicted,
            Vec::new(),
            Instant::now() + Duration::from_mins(1),
        );
        let mut pending = state
            .plan_maintenance(
                &request,
                &phase(),
                [9; 32],
                zero_memory(),
                ResidentValidationDeadline::new(request.deadline, request.deadline),
            )
            .unwrap();
        state.reserve_maintenance(&pending, None).unwrap();
        assert!(state.commit_maintenance(&mut pending).is_empty());
        assert_eq!(state.live_counts(), (0, 0));

        let next_request = fresh_request_with_prefix(Vec::new());
        let next = plan_wave_for_test(&state, &next_request, [9; 32]).unwrap();
        assert!(next
            .destinations
            .iter()
            .all(|destination| destination.generation == 2));
    }

    #[test]
    fn one_slot_refresh_only_requires_full_refresh_then_release_preserves_lineage() {
        let mut state = GpuBabBoundResidentDomainState::default();
        let mut one_slot = policy();
        one_slot.maximum_slots = 1;
        state.ensure_policy(one_slot, None, None).unwrap();
        let initial_request = single_domain_fresh_request();
        let mut initial = plan_wave_for_test(&state, &initial_request, [9; 32]).unwrap();
        state.reserve_accepted(&initial, None).unwrap();
        let mut initial_tokens = state.commit_completed(&mut initial).destination_tokens;
        let initial_token = initial_tokens.pop().unwrap();
        assert_eq!(initial_token.generation(), 1);

        let deadline = Instant::now() + Duration::from_mins(1);
        let evict_request =
            GpuBabBoundResidentMaintenanceRequest::new(Vec::new(), vec![initial_token], deadline);
        let mut evict = state
            .plan_maintenance(
                &evict_request,
                &phase(),
                [9; 32],
                zero_memory(),
                ResidentValidationDeadline::new(deadline, deadline),
            )
            .unwrap();
        state.reserve_maintenance(&evict, None).unwrap();
        let mut refresh_tokens = state.commit_maintenance(&mut evict);
        assert_eq!(state.live_counts(), (0, 1));
        let refresh_token = refresh_tokens.pop().unwrap();

        let parent_identity = *refresh_token.logical_domain_identity_sha256();
        let mut delta_wave = single_domain_fresh_request().wave;
        delta_wave.parent_groups[0].parent_identity_sha256 = parent_identity;
        let delta_request = GpuBabBoundResidentWaveRequest::new(
            delta_wave,
            history_arena_from_slice(&literal(7, 3, GpuBabBoundSplitHistoryPhase::Inactive)),
            vec![GpuBabBoundResidentParentGroup {
                parent_group_id: 10,
                prefix: GpuBabBoundArenaRange { start: 0, len: 4 },
                construction: GpuBabBoundResidentConstruction::FreshReplace,
                source: GpuBabBoundResidentParentSource::RetainedDelta {
                    parent: refresh_token,
                },
            }],
            vec![GpuBabBoundSplitHistoryView {
                suffix: GpuBabBoundArenaRange { start: 4, len: 0 },
                branch_pattern: 0,
            }],
            Vec::new(),
            Vec::new(),
        );
        let authority_deadline = ResidentValidationDeadline::new(
            delta_request.wave.deadline,
            delta_request.wave.deadline,
        );
        assert!(state
            .prevalidate_source_authority(&delta_request, [9; 32], authority_deadline)
            .unwrap());
        let (_, _, mut groups, _, _, _) = delta_request.into_parts();
        let refresh_token = match groups.pop().unwrap().source {
            GpuBabBoundResidentParentSource::RetainedDelta { parent } => parent,
            GpuBabBoundResidentParentSource::FreshUpload { .. } => {
                panic!("delta request must retain its RefreshOnly source token")
            }
        };

        let release_deadline = Instant::now() + Duration::from_mins(1);
        let release_request = GpuBabBoundResidentMaintenanceRequest::new(
            vec![refresh_token],
            Vec::new(),
            release_deadline,
        );
        let mut release = state
            .plan_maintenance(
                &release_request,
                &phase(),
                [9; 32],
                zero_memory(),
                ResidentValidationDeadline::new(release_deadline, release_deadline),
            )
            .unwrap();
        state.reserve_maintenance(&release, None).unwrap();
        assert!(state.commit_maintenance(&mut release).is_empty());
        assert_eq!(state.live_counts(), (0, 0));
        assert!(matches!(
            state.slots[0],
            GpuBabBoundResidentSlotState::Vacant { high_generation: 1 }
        ));

        let fresh_again = single_domain_fresh_request();
        let next = plan_wave_for_test(&state, &fresh_again, [9; 32]).unwrap();
        assert_eq!(next.destinations.len(), 1);
        assert_eq!(next.destinations[0].generation, 2);
        assert!(matches!(
            fresh_again.parent_groups[0].source,
            GpuBabBoundResidentParentSource::FreshUpload { prior: None }
        ));
    }

    #[test]
    fn public_one_slot_full_refresh_release_then_fresh_none_issues_generation_two() {
        let (mut lease, initial_token, maintenance_calls, resident_calls) =
            one_slot_resident_lease();
        let evict_request = GpuBabBoundResidentMaintenanceRequest::new(
            Vec::new(),
            vec![initial_token],
            lease.phase.deadline,
        );
        let evict_capability = match lease.prepare_resident_maintenance(evict_request) {
            GpuBabBoundResidentMaintenancePreparation::Accepted(capability) => capability,
            _ => panic!("one-slot eviction should be accepted"),
        };
        let refresh_token = match evict_capability.execute_accepted() {
            GpuBabBoundResidentMaintenanceDisposition::Completed(result) => {
                let (_, _, mut tokens) = result.into_parts();
                assert_eq!(tokens.len(), 1);
                tokens.pop().unwrap()
            }
            _ => panic!("one-slot eviction should complete"),
        };
        assert_eq!(lease.resident_domains.live_counts(), (0, 1));

        let parent_identity = *refresh_token.logical_domain_identity_sha256();
        let mut delta_wave = single_domain_fresh_request().wave;
        delta_wave.parent_groups[0].parent_identity_sha256 = parent_identity;
        let delta_request = GpuBabBoundResidentWaveRequest::new(
            delta_wave,
            history_arena_from_slice(&literal(7, 3, GpuBabBoundSplitHistoryPhase::Inactive)),
            vec![GpuBabBoundResidentParentGroup {
                parent_group_id: 10,
                prefix: GpuBabBoundArenaRange { start: 0, len: 4 },
                construction: GpuBabBoundResidentConstruction::FreshReplace,
                source: GpuBabBoundResidentParentSource::RetainedDelta {
                    parent: refresh_token,
                },
            }],
            vec![GpuBabBoundSplitHistoryView {
                suffix: GpuBabBoundArenaRange { start: 4, len: 0 },
                branch_pattern: 0,
            }],
            Vec::new(),
            Vec::new(),
        );
        let returned_delta = match lease.prepare_resident_wave(delta_request) {
            GpuBabBoundResidentWavePreparation::CleanDecline { reason, request } => {
                assert_eq!(reason, GpuBabBoundResidentWaveDecline::FullRefreshRequired);
                request
            }
            _ => panic!("RefreshOnly delta must return exact FullRefreshRequired request"),
        };
        let (_, _, mut groups, _, _, _) = returned_delta.into_parts();
        let refresh_token = match groups.pop().unwrap().source {
            GpuBabBoundResidentParentSource::RetainedDelta { parent } => parent,
            GpuBabBoundResidentParentSource::FreshUpload { .. } => {
                panic!("FullRefreshRequired must return the exact delta token")
            }
        };

        let release_request = GpuBabBoundResidentMaintenanceRequest::new(
            vec![refresh_token],
            Vec::new(),
            lease.phase.deadline,
        );
        let release_capability = match lease.prepare_resident_maintenance(release_request) {
            GpuBabBoundResidentMaintenancePreparation::Accepted(capability) => capability,
            _ => panic!("RefreshOnly release should be accepted"),
        };
        assert!(matches!(
            release_capability.execute_accepted(),
            GpuBabBoundResidentMaintenanceDisposition::Completed(_)
        ));
        assert_eq!(lease.resident_domains.live_counts(), (0, 0));

        let fresh_again = single_domain_fresh_request();
        assert!(matches!(
            fresh_again.parent_groups[0].source,
            GpuBabBoundResidentParentSource::FreshUpload { prior: None }
        ));
        let fresh_capability = match lease.prepare_resident_wave(fresh_again) {
            GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
            _ => panic!("Fresh prior=None should reuse the released one-slot ledger"),
        };
        let fresh_result = match fresh_capability.execute_accepted() {
            GpuBabBoundResidentWaveDisposition::Completed(result) => result,
            _ => panic!("fresh one-slot reset should complete"),
        };
        assert_eq!(fresh_result.destination_slots().len(), 1);
        assert_eq!(fresh_result.destination_slots()[0].slot_index(), 0);
        assert_eq!(fresh_result.destination_slots()[0].generation(), 2);
        assert_eq!(maintenance_calls.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(resident_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(lease.resident_domains.live_counts(), (1, 0));
        assert_eq!(lease.state, LeaseState::Open);
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::HealthyKnown
        );
    }

    #[test]
    fn stale_cross_session_and_aba_tokens_are_invalid_before_allocation() {
        let (mut state, tokens) = full_two_slot_state();
        let token = &tokens[0];
        let cross_session = GpuBabBoundResidentSlotRef {
            session_nonce_sha256: [8; 32],
            logical_domain_identity_sha256: *token.logical_domain_identity_sha256(),
            slot_index: token.slot_index(),
            generation: token.generation(),
        };
        let stale_generation = GpuBabBoundResidentSlotRef {
            session_nonce_sha256: *token.session_nonce_sha256(),
            logical_domain_identity_sha256: *token.logical_domain_identity_sha256(),
            slot_index: token.slot_index(),
            generation: token.generation() + 1,
        };
        for forged in [cross_session, stale_generation] {
            let deadline = Instant::now() + Duration::from_mins(1);
            let request =
                GpuBabBoundResidentMaintenanceRequest::new(vec![forged], Vec::new(), deadline);
            let _allocation = inject_resident_allocation_failure(0);
            assert!(matches!(
                state.plan_maintenance(
                    &request,
                    &phase(),
                    [9; 32],
                    zero_memory(),
                    ResidentValidationDeadline::new(deadline, deadline),
                ),
                Err(GpuBabBoundResidentAdmissionError::Invalid(_))
            ));
        }
        assert_eq!(state.live_counts(), (2, 0));
        assert!(state.resources_are_quiescent());

        let old = GpuBabBoundResidentSlotRef {
            session_nonce_sha256: *tokens[0].session_nonce_sha256(),
            logical_domain_identity_sha256: *tokens[0].logical_domain_identity_sha256(),
            slot_index: tokens[0].slot_index(),
            generation: tokens[0].generation(),
        };
        let deadline = Instant::now() + Duration::from_mins(1);
        let release_request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), deadline);
        let mut release = state
            .plan_maintenance(
                &release_request,
                &phase(),
                [9; 32],
                zero_memory(),
                ResidentValidationDeadline::new(deadline, deadline),
            )
            .unwrap();
        state.reserve_maintenance(&release, None).unwrap();
        state.commit_maintenance(&mut release);
        let fresh = fresh_request_with_prefix(Vec::new());
        let mut replacement = plan_wave_for_test(&state, &fresh, [9; 32]).unwrap();
        state.reserve_accepted(&replacement, None).unwrap();
        let replacements = state.commit_completed(&mut replacement).destination_tokens;
        assert!(replacements.iter().all(|token| token.generation() == 2));
        let deadline = Instant::now() + Duration::from_mins(1);
        let aba_request =
            GpuBabBoundResidentMaintenanceRequest::new(vec![old], Vec::new(), deadline);
        assert!(matches!(
            state.plan_maintenance(
                &aba_request,
                &phase(),
                [9; 32],
                zero_memory(),
                ResidentValidationDeadline::new(deadline, deadline),
            ),
            Err(GpuBabBoundResidentAdmissionError::Invalid(_))
        ));
    }

    #[test]
    fn maintenance_rollback_preserves_sources_and_receipt_rejects_every_work_class() {
        let (mut state, tokens) = full_two_slot_state();
        let deadline = Instant::now() + Duration::from_mins(1);
        let request = GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), deadline);
        let pending = state
            .plan_maintenance(
                &request,
                &phase(),
                [9; 32],
                zero_memory(),
                ResidentValidationDeadline::new(request.deadline, request.deadline),
            )
            .unwrap();
        state.reserve_maintenance(&pending, None).unwrap();
        let transcript = maintenance_transcript(&pending, deadline);
        let receipt =
            core_resident_maintenance_receipt(&pending, transcript, zero_memory(), false).unwrap();
        validate_resident_maintenance_receipt(
            &receipt,
            &pending,
            transcript,
            zero_memory(),
            1 << 20,
            false,
        )
        .unwrap();
        for mutation in 0..14 {
            let mut corrupt = receipt;
            match mutation {
                0 => corrupt.memory.destination_slots = 1,
                1 => corrupt.memory.allocated_destination_slots = 1,
                2 => corrupt.memory.destination_buffer_units = 1,
                3 => corrupt.memory.allocated_destination_buffer_units = 1,
                4 => corrupt.memory.allocated_destination_bytes = 1,
                5 => corrupt.memory.destination_padding_bytes = 1,
                6 => corrupt.host_to_device_bytes = 1,
                7 => corrupt.device_to_host_bytes = 1,
                8 => corrupt.device_to_device_bytes = 1,
                9 => corrupt.control_payload_bytes = 1,
                10 => corrupt.transfer_units = 1,
                11 => corrupt.completed_transfer_units = 1,
                12 => corrupt.dispatches = 1,
                13 => corrupt.submits = 1,
                _ => unreachable!(),
            }
            assert!(validate_resident_maintenance_receipt(
                &corrupt,
                &pending,
                transcript,
                zero_memory(),
                1 << 20,
                false,
            )
            .is_err());
        }
        state.rollback_maintenance(&pending);
        assert_eq!(state.live_counts(), (2, 0));
        assert!(state.slots.iter().all(|slot| matches!(
            slot,
            GpuBabBoundResidentSlotState::Live(GpuBabBoundResidentLiveSlot {
                in_flight: false,
                ..
            })
        )));
    }

    #[test]
    fn maintenance_runs_at_exact_core_host_cap_without_new_headroom() {
        let (mut state, tokens) = full_two_slot_state();
        let exact_host = state.core_host_charged_bytes().unwrap();
        let GpuBabBoundResidentPolicyState::Installed(mut installed) = state.policy_state else {
            unreachable!()
        };
        installed.maximum_retained_v2_core_host_charged_bytes = exact_host;
        assert!(installed.is_valid());
        state.policy_state = GpuBabBoundResidentPolicyState::Installed(installed);
        let request = GpuBabBoundResidentMaintenanceRequest::new(
            tokens,
            Vec::new(),
            Instant::now() + Duration::from_mins(1),
        );
        let pending = state
            .plan_maintenance(
                &request,
                &phase(),
                [9; 32],
                zero_memory(),
                ResidentValidationDeadline::new(request.deadline, request.deadline),
            )
            .unwrap();
        assert_eq!(
            pending
                .host_audit
                .retained_v2_core_host_before_charged_bytes,
            exact_host
        );
        assert_eq!(
            pending.host_audit.retained_v2_core_host_peak_charged_bytes,
            exact_host
        );
        assert_eq!(
            pending.host_audit.retained_v2_core_host_after_charged_bytes,
            2 * GPU_BAB_BOUND_HOST_CONFIGURED_SLOT_RESERVE_BYTES
        );
    }

    #[test]
    fn maintenance_retry_returns_owned_tokens_and_never_permits_legacy_fallback() {
        let (mut lease, tokens, execute_calls) = maintenance_lease(MaintenancePrepareMode::Retry);
        let expected_slots: Vec<_> = tokens
            .iter()
            .map(GpuBabBoundResidentSlotRef::slot_index)
            .collect();
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), lease.phase.deadline);
        let (reason, request) = {
            let preparation = lease.prepare_resident_maintenance(request);
            assert!(!preparation.permits_legacy_fallback());
            assert!(preparation.permits_retry());
            match preparation {
                GpuBabBoundResidentMaintenancePreparation::CleanDecline { reason, request } => {
                    (reason, request)
                }
                _ => panic!("maintenance retry must return the exact owned request"),
            }
        };
        assert_eq!(
            reason,
            GpuBabBoundResidentWaveDecline::TemporarilyUnavailable
        );
        assert_eq!(
            request
                .release()
                .iter()
                .map(GpuBabBoundResidentSlotRef::slot_index)
                .collect::<Vec<_>>(),
            expected_slots
        );
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(lease.resident_domains.live_counts(), (2, 0));
    }

    #[test]
    fn accepted_maintenance_completed_is_atomic_and_reenables_v1_at_zero_residency() {
        let (mut lease, tokens, execute_calls) =
            maintenance_lease(MaintenancePrepareMode::Accepted);
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), lease.phase.deadline);
        let capability = match lease.prepare_resident_maintenance(request) {
            GpuBabBoundResidentMaintenancePreparation::Accepted(capability) => capability,
            _ => panic!("maintenance should be accepted"),
        };
        let result = match capability.execute_accepted() {
            GpuBabBoundResidentMaintenanceDisposition::Completed(result) => result,
            _ => panic!("maintenance completion should validate"),
        };
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(result.receipt().memory.resident_device_after_bytes, 0);
        assert_eq!(result.host_audit().history_after_words, 0);
        assert_eq!(lease.resident_domains.live_counts(), (0, 0));
        assert_eq!(lease.resident_domains.resident_bytes().unwrap(), 0);
        assert!(matches!(
            lease.prepare_wave(base_wave()),
            GpuBabBoundWavePreparation::CleanDecline(
                GpuBabBoundWaveDecline::TemporarilyUnavailable
            )
        ));
    }

    #[test]
    fn dropping_accepted_maintenance_rolls_back_then_absorbingly_poisons() {
        let (mut lease, tokens, execute_calls) =
            maintenance_lease(MaintenancePrepareMode::Accepted);
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), lease.phase.deadline);
        let capability = match lease.prepare_resident_maintenance(request) {
            GpuBabBoundResidentMaintenancePreparation::Accepted(capability) => capability,
            _ => panic!("maintenance should be accepted"),
        };
        drop(capability);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert!(lease.resident_domains.poisoned);
        assert_eq!(lease.resident_domains.live_counts(), (2, 0));
        let terminal = lease
            .abandoned_resident_maintenance_terminal()
            .expect("abandoned maintenance records a core terminal");
        assert_eq!(
            terminal.kind(),
            GpuBabBoundTerminalFailureKind::CapabilityAbandoned
        );
        assert!(terminal.receipt_validated());
        assert!(matches!(
            lease.prepare_resident_maintenance(GpuBabBoundResidentMaintenanceRequest::new(
                Vec::new(),
                Vec::new(),
                Instant::now() + Duration::from_mins(1),
            )),
            GpuBabBoundResidentMaintenancePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::PoisonedOrBusy
            )
        ));
    }

    #[test]
    fn validation_deadline_injection_is_labeled_scoped_and_v1_opted_out() {
        let real_deadline = Instant::now() + Duration::from_mins(1);
        {
            let _outer = inject_resident_validation_deadline("outer validation label", 0);
            let v1_deadline = ResidentValidationDeadline::new_without_test_injection(
                real_deadline,
                real_deadline,
            );
            assert!(!v1_deadline.expired("outer validation label"));
            {
                let _inner = inject_resident_validation_deadline("inner validation label", 0);
                let v2_deadline = ResidentValidationDeadline::new(real_deadline, real_deadline);
                assert!(v2_deadline.check("inner validation label").is_err());
                assert!(v2_deadline.check("inner validation label").is_ok());
            }
            let v2_deadline = ResidentValidationDeadline::new(real_deadline, real_deadline);
            assert!(v2_deadline.check("outer validation label").is_err());
        }
        let v2_deadline = ResidentValidationDeadline::new(real_deadline, real_deadline);
        assert!(v2_deadline.check("outer validation label").is_ok());
    }

    #[test]
    fn nominal_budget_rejects_first_observed_overcapacity_without_changing_charge() {
        let mut budget = ResidentHostAdmissionBudget::new(100, 0, 60, 40).unwrap();
        assert!(budget.charge_metadata_capacity(1, 2, 1).is_err());
        assert_eq!(budget.base_charged_bytes, 0);
        assert_eq!(budget.metadata_charged_bytes, 60);
        assert_eq!(budget.snapshot_charged_bytes, 40);
    }

    #[test]
    fn injected_wave_allocation_failure_returns_exact_request_before_raw_or_slot_install() {
        let phase = phase();
        let (mut lease, prepare_calls, execute_calls) = resident_admission_lease(phase);
        let request = fresh_request_with_prefix(Vec::new());
        let mut expected = fresh_request_with_prefix(Vec::new());
        expected.wave.deadline = request.wave.deadline;
        let returned = {
            let _injection = inject_resident_allocation_failure(0);
            match lease.prepare_resident_wave(request) {
                GpuBabBoundResidentWavePreparation::CleanDecline { reason, request } => {
                    assert_eq!(reason, GpuBabBoundResidentWaveDecline::InsufficientCapacity);
                    request
                }
                _ => panic!("first resident allocation failure must clean-decline"),
            }
        };
        assert_eq!(returned, expected);
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert!(matches!(
            lease.resident_domains.policy_state,
            GpuBabBoundResidentPolicyState::Observed(_)
        ));
        assert!(lease.resident_domains.slots.is_empty());
        assert_eq!(lease.state, LeaseState::Open);
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::HealthyKnown
        );
    }

    #[test]
    fn foreign_maintenance_token_is_invalid_before_uninstalled_slot_allocation() {
        let phase = phase();
        let deadline = phase.deadline;
        let (mut lease, _, _, _, prepare_calls, execute_calls) =
            resident_admission_lease_with_policy(
                phase,
                ResidentExecuteMode::Completed,
                Some(policy()),
                None,
                None,
            );
        let foreign = GpuBabBoundResidentSlotRef {
            session_nonce_sha256: [0xA5; 32],
            logical_domain_identity_sha256: [0x5A; 32],
            slot_index: 0,
            generation: 1,
        };
        let expected = GpuBabBoundResidentMaintenanceRequest::new(
            vec![GpuBabBoundResidentSlotRef {
                session_nonce_sha256: [0xA5; 32],
                logical_domain_identity_sha256: [0x5A; 32],
                slot_index: 0,
                generation: 1,
            }],
            Vec::new(),
            deadline,
        );
        let returned =
            {
                let _injection = inject_resident_allocation_failure(0);
                match lease.prepare_resident_maintenance(
                    GpuBabBoundResidentMaintenanceRequest::new(vec![foreign], Vec::new(), deadline),
                ) {
                    GpuBabBoundResidentMaintenancePreparation::InvalidRequest {
                        request, ..
                    } => request,
                    _ => panic!("foreign maintenance authority must win before slot allocation"),
                }
            };
        assert_eq!(returned, expected);
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert!(matches!(
            lease.resident_domains.policy_state,
            GpuBabBoundResidentPolicyState::Observed(_)
        ));
        assert!(lease.resident_domains.slots.is_empty());
        assert_eq!(lease.state, LeaseState::Open);
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::HealthyKnown
        );
    }

    #[test]
    fn injected_maintenance_allocation_failure_returns_all_tokens_before_raw() {
        let slot_count = VALIDATION_POLL_STRIDE + 1;
        let (mut lease, tokens, prepare_calls, execute_calls) = large_maintenance_lease(slot_count);
        let first = (
            *tokens[0].session_nonce_sha256(),
            *tokens[0].logical_domain_identity_sha256(),
            tokens[0].slot_index(),
            tokens[0].generation(),
        );
        let last = (
            *tokens[slot_count - 1].session_nonce_sha256(),
            *tokens[slot_count - 1].logical_domain_identity_sha256(),
            tokens[slot_count - 1].slot_index(),
            tokens[slot_count - 1].generation(),
        );
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), lease.phase.deadline);
        let returned = {
            let _injection = inject_resident_allocation_failure(0);
            match lease.prepare_resident_maintenance(request) {
                GpuBabBoundResidentMaintenancePreparation::CleanDecline { reason, request } => {
                    assert_eq!(
                        reason,
                        GpuBabBoundResidentWaveDecline::TemporarilyUnavailable
                    );
                    request
                }
                _ => panic!("first maintenance allocation failure must clean-decline"),
            }
        };
        assert_eq!(returned.release().len(), slot_count);
        assert!(returned.evict().is_empty());
        let returned_first = &returned.release()[0];
        let returned_last = &returned.release()[slot_count - 1];
        assert_eq!(
            (
                *returned_first.session_nonce_sha256(),
                *returned_first.logical_domain_identity_sha256(),
                returned_first.slot_index(),
                returned_first.generation(),
            ),
            first
        );
        assert_eq!(
            (
                *returned_last.session_nonce_sha256(),
                *returned_last.logical_domain_identity_sha256(),
                returned_last.slot_index(),
                returned_last.generation(),
            ),
            last
        );
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(lease.resident_domains.live_counts(), (slot_count, 0));
        assert!(lease.resident_domains.resources_are_quiescent());
        assert_eq!(lease.state, LeaseState::Open);
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::HealthyKnown
        );
    }

    #[test]
    fn injected_mid_base_arena_deadline_stops_before_raw_preflight() {
        let phase = phase();
        let (mut lease, resident_prepare_calls, _) = resident_admission_lease(phase);
        let request = large_activation_request();
        let terminal = {
            let _injection =
                inject_resident_validation_deadline("resident f32 arena validation", 1);
            match lease.prepare_resident_wave(request) {
                GpuBabBoundResidentWavePreparation::SessionTerminal(terminal) => terminal,
                _ => panic!("mid-arena deadline must terminalize before raw preflight"),
            }
        };
        assert_eq!(
            terminal,
            GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired
        );
        assert_eq!(resident_prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(lease.state, LeaseState::Poisoned);
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
        assert!(lease.resident_domains.resources_are_quiescent());
        assert!(matches!(
            lease.resident_domains.policy_state,
            GpuBabBoundResidentPolicyState::Observed(_)
        ));
        assert!(lease.resident_domains.slots.is_empty());
    }

    #[test]
    fn injected_mid_history_copy_drops_partial_plan_before_raw_preflight() {
        let phase = phase();
        let (mut lease, resident_prepare_calls, _) = resident_admission_lease(phase);
        let record_count = VALIDATION_POLL_STRIDE / GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS + 1;
        let mut prefix =
            Vec::with_capacity(record_count * GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS);
        for index in 0..record_count {
            prefix.extend(literal(
                100 + index as u32,
                index as u32,
                GpuBabBoundSplitHistoryPhase::Inactive,
            ));
        }
        let request = fresh_request_with_prefix(prefix);
        RESIDENT_SNAPSHOT_MATERIALIZATIONS.store(0, Ordering::Relaxed);
        let terminal = {
            let _injection =
                inject_resident_validation_deadline("resident compact history copy", 1);
            match lease.prepare_resident_wave(request) {
                GpuBabBoundResidentWavePreparation::SessionTerminal(terminal) => terminal,
                _ => panic!("mid-copy deadline must terminalize before raw preflight"),
            }
        };
        assert_eq!(
            terminal,
            GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired
        );
        assert_eq!(resident_prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(
            RESIDENT_SNAPSHOT_MATERIALIZATIONS.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
        assert!(lease.resident_domains.resources_are_quiescent());
    }

    #[test]
    fn injected_maintenance_scan_deadline_preserves_live_tokens_before_raw() {
        let slot_count = VALIDATION_POLL_STRIDE + 1;
        let (mut lease, tokens, prepare_calls, execute_calls) = large_maintenance_lease(slot_count);
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), lease.phase.deadline);
        let terminal = {
            let _injection =
                inject_resident_validation_deadline("resident maintenance source audit", 1);
            match lease.prepare_resident_maintenance(request) {
                GpuBabBoundResidentMaintenancePreparation::SessionTerminal(terminal) => terminal,
                _ => panic!("mid-maintenance scan deadline must stop before raw preflight"),
            }
        };
        assert_eq!(
            terminal,
            GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired
        );
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(lease.resident_domains.live_counts(), (slot_count, 0));
        assert!(lease.resident_domains.resources_are_quiescent());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
    }

    #[test]
    fn injected_postreservation_deadline_rolls_back_known_without_raw_execution() {
        let (mut lease, tokens, execute_calls) =
            maintenance_lease(MaintenancePrepareMode::Accepted);
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), lease.phase.deadline);
        let failure = {
            let _injection = inject_resident_validation_deadline(
                "maintenance post-reservation deadline gate",
                0,
            );
            match lease.prepare_resident_maintenance(request) {
                GpuBabBoundResidentMaintenancePreparation::DeadlineExpired(failure) => failure,
                _ => panic!("post-reservation expiry must return a typed deadline terminal"),
            }
        };
        assert!(failure.receipt_validated());
        let audit = failure
            .host_audit()
            .expect("rollback retains exact host audit");
        assert_eq!(audit.history_after_words, audit.history_before_words);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(lease.resident_domains.live_counts(), (2, 0));
        assert!(lease.resident_domains.resources_are_quiescent());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
    }

    #[test]
    fn injected_maintenance_postcommit_deadline_withholds_tokens_and_keeps_commit_known() {
        let (mut lease, tokens, execute_calls) =
            maintenance_lease(MaintenancePrepareMode::Accepted);
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), lease.phase.deadline);
        let capability = match lease.prepare_resident_maintenance(request) {
            GpuBabBoundResidentMaintenancePreparation::Accepted(capability) => capability,
            _ => panic!("maintenance should be accepted before postcommit injection"),
        };
        let _settlement_probe = inject_resident_completed_settlement_probe();
        let failure = {
            let _injection =
                inject_resident_validation_deadline("maintenance postcommit deadline gate", 0);
            match capability.execute_accepted() {
                GpuBabBoundResidentMaintenanceDisposition::DeadlineExpired(failure) => failure,
                _ => panic!("postcommit expiry must withhold the completed result"),
            }
        };
        assert_eq!(resident_completed_settlement_probe_counts(), (1, 0));
        assert!(failure.receipt_validated());
        let audit = failure
            .host_audit()
            .expect("completed commit has an exact audit");
        assert_eq!(audit.history_after_words, 0);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(lease.resident_domains.live_counts(), (0, 0));
        assert!(lease.resident_domains.resources_are_quiescent());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
    }

    #[test]
    fn malformed_completed_maintenance_receipt_settles_unknown_before_diagnostic() {
        let (mut lease, tokens, execute_calls) =
            maintenance_lease_with_receipt_mode(MaintenancePrepareMode::Accepted, true);
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), lease.phase.deadline);
        let capability = match lease.prepare_resident_maintenance(request) {
            GpuBabBoundResidentMaintenancePreparation::Accepted(capability) => capability,
            _ => panic!("maintenance should be accepted before malformed completion"),
        };
        let _settlement_probe = inject_resident_completed_settlement_probe();
        let failure = match capability.execute_accepted() {
            GpuBabBoundResidentMaintenanceDisposition::AcceptedFailure(failure) => failure,
            _ => panic!("malformed maintenance completion must be quarantined"),
        };
        assert_eq!(resident_completed_settlement_probe_counts(), (1, 0));
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(!failure.receipt_validated());
        assert!(failure.host_audit().is_some());
        assert_eq!(lease.resident_domains.live_counts(), (2, 0));
        assert_eq!(lease.resident_domains.in_flight_slots, 2);
        assert!(!lease.resident_domains.resources_are_quiescent());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        match lease.close() {
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                receipt_validated, ..
            } => assert!(!receipt_validated),
            GpuBabBoundPhaseCloseDisposition::Closed(_) => {
                panic!("malformed maintenance completion must not close as validated")
            }
        }
    }

    #[test]
    fn injected_wave_reserve_panic_is_quarantined_unknown_without_publication() {
        let phase = phase();
        let (mut lease, resident_prepare_calls, _) = resident_admission_lease(phase);
        let failure = {
            let _injection = inject_resident_journal_panic("resident reserve mutation", 1);
            match lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                GpuBabBoundResidentWavePreparation::AcceptedFailure(failure) => failure,
                _ => panic!("mid-reserve panic must become an unvalidated accepted failure"),
            }
        };
        assert_eq!(resident_prepare_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(!failure.receipt_validated());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        assert_eq!(lease.state, LeaseState::Poisoned);
        assert!(lease.resident_domains.poisoned);
        assert!(lease
            .resident_domains
            .ledger_audit_with_deadline(None)
            .is_err());
    }

    #[test]
    fn injected_wave_rollback_panic_is_never_retried_or_labeled_known() {
        let phase = phase();
        let (mut lease, resident_prepare_calls, _) = resident_admission_lease(phase);
        let failure = {
            let _rollback = inject_resident_journal_panic("resident rollback mutation", 1);
            let _deadline =
                inject_resident_validation_deadline("resident post-reservation deadline gate", 0);
            match lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                GpuBabBoundResidentWavePreparation::DeadlineExpired(failure) => failure,
                _ => panic!("rollback unwind after reservation must stay a deadline terminal"),
            }
        };
        assert_eq!(resident_prepare_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(!failure.receipt_validated());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        assert!(lease
            .resident_domains
            .ledger_audit_with_deadline(None)
            .is_err());
    }

    #[test]
    fn injected_maintenance_reserve_and_rollback_panics_are_quarantined() {
        let (mut reserve_lease, tokens, reserve_execute_calls) =
            maintenance_lease(MaintenancePrepareMode::Accepted);
        let request = GpuBabBoundResidentMaintenanceRequest::new(
            tokens,
            Vec::new(),
            reserve_lease.phase.deadline,
        );
        let reserve_failure = {
            let _injection = inject_resident_journal_panic("maintenance reserve mutation", 1);
            match reserve_lease.prepare_resident_maintenance(request) {
                GpuBabBoundResidentMaintenancePreparation::AcceptedFailure(failure) => failure,
                _ => panic!("maintenance reserve panic must be caught"),
            }
        };
        assert!(!reserve_failure.receipt_validated());
        assert_eq!(reserve_execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(
            reserve_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );

        let (mut rollback_lease, tokens, rollback_execute_calls) =
            maintenance_lease(MaintenancePrepareMode::Accepted);
        let request = GpuBabBoundResidentMaintenanceRequest::new(
            tokens,
            Vec::new(),
            rollback_lease.phase.deadline,
        );
        let rollback_failure = {
            let _rollback = inject_resident_journal_panic("maintenance rollback mutation", 1);
            let _deadline = inject_resident_validation_deadline(
                "maintenance post-reservation deadline gate",
                0,
            );
            match rollback_lease.prepare_resident_maintenance(request) {
                GpuBabBoundResidentMaintenancePreparation::DeadlineExpired(failure) => failure,
                _ => panic!("maintenance rollback panic must be caught"),
            }
        };
        assert!(!rollback_failure.receipt_validated());
        assert_eq!(rollback_execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(
            rollback_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
    }

    #[test]
    fn injected_maintenance_commit_panic_has_no_host_audit_or_tokens() {
        let (mut lease, tokens, execute_calls) =
            maintenance_lease(MaintenancePrepareMode::Accepted);
        let request =
            GpuBabBoundResidentMaintenanceRequest::new(tokens, Vec::new(), lease.phase.deadline);
        let capability = match lease.prepare_resident_maintenance(request) {
            GpuBabBoundResidentMaintenancePreparation::Accepted(capability) => capability,
            _ => panic!("maintenance should be accepted before commit injection"),
        };
        let failure = {
            let _injection = inject_resident_journal_panic("maintenance commit mutation", 1);
            match capability.execute_accepted() {
                GpuBabBoundResidentMaintenanceDisposition::AcceptedFailure(failure) => failure,
                _ => panic!("commit panic must become an unvalidated failure"),
            }
        };
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(!failure.receipt_validated());
        assert!(failure.host_audit().is_none());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        assert_eq!(lease.state, LeaseState::Poisoned);
    }

    #[test]
    fn injected_wave_commit_panic_is_caught_before_any_token_publication() {
        let mut state = GpuBabBoundResidentDomainState::default();
        state.ensure_policy(policy(), None, None).unwrap();
        let request = fresh_request_with_prefix(Vec::new());
        let mut pending = plan_wave_for_test(&state, &request, [9; 32]).unwrap();
        state.reserve_accepted(&pending, None).unwrap();
        let token_count = pending.destination_tokens.len();
        let commit = {
            let _injection = inject_resident_journal_panic("resident commit mutation", 1);
            guarded_commit_resident(&mut state, &mut pending)
        };
        assert!(commit.is_err());
        state.poison_all();
        assert_eq!(pending.destination_tokens.len(), token_count);
        assert!(state.poisoned);
        assert!(state.ledger_audit_with_deadline(None).is_err());
    }

    #[test]
    fn injected_resident_result_scan_deadline_commits_known_and_withholds_outputs() {
        let phase = phase();
        let (mut lease, prepare_calls, execute_calls) = resident_admission_lease(phase);
        let capability = match lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
            GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
            _ => panic!("resident wave should be accepted before result-scan injection"),
        };
        let _settlement_probe = inject_resident_completed_settlement_probe();
        let failure = {
            let _injection =
                inject_resident_validation_deadline("resident completed row validation", 0);
            match capability.execute_accepted() {
                GpuBabBoundResidentWaveDisposition::DeadlineExpired(failure) => failure,
                _ => panic!("result-scan deadline must commit and withhold public outputs"),
            }
        };
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(resident_completed_settlement_probe_counts(), (1, 0));
        assert!(failure.receipt_validated());
        assert!(failure
            .host_audit()
            .is_some_and(|audit| audit.history_after_words > 0));
        assert_eq!(lease.resident_domains.live_counts(), (2, 0));
        assert!(lease.resident_domains.resources_are_quiescent());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
    }

    #[test]
    fn injected_resident_pre_and_postcommit_deadlines_both_withhold_tokens() {
        for label in [
            "resident precommit deadline gate",
            "resident postcommit deadline gate",
        ] {
            let phase = phase();
            let (mut lease, prepare_calls, execute_calls) = resident_admission_lease(phase);
            let capability =
                match lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                    GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
                    _ => panic!("resident wave should be accepted before commit injection"),
                };
            let failure = {
                let _injection = inject_resident_validation_deadline(label, 0);
                match capability.execute_accepted() {
                    GpuBabBoundResidentWaveDisposition::DeadlineExpired(failure) => failure,
                    _ => panic!("commit-boundary deadline must withhold tokens"),
                }
            };
            assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 1);
            assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
            assert!(failure.receipt_validated());
            assert!(failure
                .host_audit()
                .is_some_and(|audit| audit.history_after_words > 0));
            assert_eq!(lease.resident_domains.live_counts(), (2, 0));
            assert!(lease.resident_domains.resources_are_quiescent());
            assert_eq!(
                lease.resource_certainty,
                ResidentResourceCertainty::PoisonedKnown
            );
        }
    }

    #[test]
    fn injected_resident_commit_panic_is_unknown_with_no_host_audit_or_tokens() {
        let phase = phase();
        let (mut lease, prepare_calls, execute_calls) = resident_admission_lease(phase);
        let capability = match lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
            GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
            _ => panic!("resident wave should be accepted before commit panic injection"),
        };
        let failure = {
            let _injection = inject_resident_journal_panic("resident commit mutation", 1);
            match capability.execute_accepted() {
                GpuBabBoundResidentWaveDisposition::AcceptedFailure(failure) => failure,
                _ => panic!("resident commit panic must be quarantined"),
            }
        };
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(!failure.receipt_validated());
        assert!(failure.host_audit().is_none());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        assert_eq!(lease.state, LeaseState::Poisoned);
    }

    #[test]
    fn completed_resident_wave_publishes_canonical_tokens_and_keeps_healthy_authority() {
        let phase = phase();
        let (mut lease, prepare_calls, execute_calls) = resident_admission_lease(phase);
        let result = match lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
            GpuBabBoundResidentWavePreparation::Accepted(capability) => {
                match capability.execute_accepted() {
                    GpuBabBoundResidentWaveDisposition::Completed(result) => result,
                    _ => panic!("exact resident completion must publish a validated result"),
                }
            }
            _ => panic!("fresh resident wave should be accepted"),
        };
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(result.domain_outcomes().len(), 2);
        assert_eq!(result.rows().len(), 4);
        assert_eq!(result.destination_slots().len(), 2);
        assert!(result.evicted_slots().is_empty());
        for (slot_index, token) in result.destination_slots().iter().enumerate() {
            assert_eq!(
                token.session_nonce_sha256(),
                &lease.transcript.backend.session_nonce_sha256
            );
            assert_eq!(token.slot_index(), slot_index as u32);
            assert_eq!(token.generation(), 1);
        }
        assert_eq!(lease.resident_domains.live_counts(), (2, 0));
        assert!(lease.resident_domains.resources_are_quiescent());
        assert_eq!(lease.state, LeaseState::Open);
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::HealthyKnown
        );
    }

    #[test]
    fn malformed_completed_result_stays_reserved_unknown_and_close_is_unvalidated() {
        let phase = phase();
        let (mut lease, prepare_calls, execute_calls) =
            resident_admission_lease_with_mode(phase, ResidentExecuteMode::MalformedRow);
        let _settlement_probe = inject_resident_completed_settlement_probe();
        let failure = match lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
            GpuBabBoundResidentWavePreparation::Accepted(capability) => {
                match capability.execute_accepted() {
                    GpuBabBoundResidentWaveDisposition::AcceptedFailure(failure) => failure,
                    _ => panic!("malformed completed association must be rejected"),
                }
            }
            _ => panic!("fresh resident wave should be accepted"),
        };
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(resident_completed_settlement_probe_counts(), (1, 0));
        assert!(!failure.receipt_validated());
        assert!(failure.host_audit().is_some());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        assert_eq!(lease.state, LeaseState::Poisoned);
        assert!(!lease.resident_domains.resources_are_quiescent());
        assert_eq!(lease.resident_domains.reserved_slots, 2);
        match lease.close() {
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                receipt_validated, ..
            } => assert!(!receipt_validated),
            GpuBabBoundPhaseCloseDisposition::Closed(_) => {
                panic!("unknown retained authority must not close as validated")
            }
        }
    }

    #[test]
    fn known_poisoned_postcommit_deadline_has_exact_but_nonreleasing_close() {
        let phase = phase();
        let (mut lease, _, _) = resident_admission_lease(phase);
        let capability = match lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
            GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
            _ => panic!("resident wave should be accepted"),
        };
        let failure = {
            let _injection =
                inject_resident_validation_deadline("resident postcommit deadline gate", 0);
            match capability.execute_accepted() {
                GpuBabBoundResidentWaveDisposition::DeadlineExpired(failure) => failure,
                _ => panic!("postcommit deadline must withhold the completed result"),
            }
        };
        assert!(failure.receipt_validated());
        assert!(failure.host_audit().is_some());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
        assert!(lease.resident_domains.resources_are_quiescent());
        match lease.close() {
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                receipt_validated,
                receipt,
                core_host_audit,
                ..
            } => {
                assert!(receipt_validated);
                assert!(receipt.is_some());
                assert!(core_host_audit.is_some());
            }
            GpuBabBoundPhaseCloseDisposition::Closed(_) => {
                panic!("known-poisoned cleanup must never release registration authority")
            }
        }
    }

    #[test]
    fn policy_changes_before_prepare_or_capability_execute_settle_known_without_raw_work() {
        let mut changed_policy = policy();
        changed_policy.maximum_history_words += 1;

        let (mut prepare_lease, prepare_calls, execute_calls, policy_calls, _, _) =
            resident_admission_lease_with_policy(
                phase(),
                ResidentExecuteMode::Completed,
                Some(policy()),
                Some(1),
                Some(changed_policy),
            );
        assert!(matches!(
            prepare_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())),
            GpuBabBoundResidentWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendResidentAuthorityLost
            )
        ));
        assert_eq!(policy_calls.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(
            prepare_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
        assert!(prepare_lease.resident_domains.resources_are_quiescent());

        let (mut capability_lease, prepare_calls, execute_calls, policy_calls, _, _) =
            resident_admission_lease_with_policy(
                phase(),
                ResidentExecuteMode::Completed,
                Some(policy()),
                Some(3),
                Some(changed_policy),
            );
        let capability =
            match capability_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
                _ => panic!("stable preflight observations should issue a capability"),
            };
        let failure = match capability.execute_accepted() {
            GpuBabBoundResidentWaveDisposition::AcceptedFailure(failure) => failure,
            _ => panic!("capability policy change must rollback without raw execution"),
        };
        assert_eq!(policy_calls.load(AtomicOrdering::Relaxed), 4);
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert!(failure.receipt_validated());
        assert_eq!(
            capability_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
        assert!(capability_lease.resident_domains.resources_are_quiescent());
    }

    #[test]
    fn registration_loss_before_capability_or_close_forces_unknown_unvalidated_cleanup() {
        let (mut capability_lease, _, execute_calls) = resident_admission_lease(phase());
        let registration = capability_lease.registration;
        let identity = capability_lease.transcript.backend;
        let capability =
            match capability_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
                _ => panic!("resident wave should be accepted before registration loss"),
            };
        registration.poison(identity);
        let failure = match capability.execute_accepted() {
            GpuBabBoundResidentWaveDisposition::AcceptedFailure(failure) => failure,
            _ => panic!("lost capability authority must return an unvalidated terminal"),
        };
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        assert!(!failure.receipt_validated());
        assert_eq!(
            capability_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        assert!(capability_lease.resident_domains.resources_are_quiescent());

        let (close_lease, _, _) = resident_admission_lease(phase());
        let registration = close_lease.registration;
        let identity = close_lease.transcript.backend;
        registration.poison(identity);
        match close_lease.close() {
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                receipt_validated, ..
            } => assert!(!receipt_validated),
            GpuBabBoundPhaseCloseDisposition::Closed(_) => {
                panic!("registration loss must prevent validated close")
            }
        }
    }

    #[test]
    fn postraw_policy_change_is_unknown_but_close_policy_change_is_exact_known_cleanup() {
        let mut changed_policy = policy();
        changed_policy.maximum_history_words += 1;
        let (mut terminal_lease, _, execute_calls, policy_calls, _, _) =
            resident_admission_lease_with_policy(
                phase(),
                ResidentExecuteMode::Completed,
                Some(policy()),
                Some(4),
                Some(changed_policy),
            );
        let capability =
            match terminal_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
                _ => panic!("resident wave should be accepted before terminal recheck"),
            };
        let failure = match capability.execute_accepted() {
            GpuBabBoundResidentWaveDisposition::AcceptedFailure(failure) => failure,
            _ => panic!("postraw policy change must quarantine the receipt"),
        };
        assert_eq!(policy_calls.load(AtomicOrdering::Relaxed), 5);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(!failure.receipt_validated());
        assert_eq!(
            terminal_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        assert!(!terminal_lease.resident_domains.resources_are_quiescent());

        let (mut close_lease, _, execute_calls, policy_calls, _, _) =
            resident_admission_lease_with_policy(
                phase(),
                ResidentExecuteMode::Completed,
                Some(policy()),
                Some(5),
                Some(changed_policy),
            );
        let capability =
            match close_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
                _ => panic!("resident wave should be accepted before close recheck"),
            };
        assert!(matches!(
            capability.execute_accepted(),
            GpuBabBoundResidentWaveDisposition::Completed(_)
        ));
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 1);
        match close_lease.close() {
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                receipt_validated, ..
            } => assert!(receipt_validated),
            GpuBabBoundPhaseCloseDisposition::Closed(_) => {
                panic!("a pure close-time policy change must poison without release")
            }
        }
        assert_eq!(policy_calls.load(AtomicOrdering::Relaxed), 6);
    }

    #[test]
    fn unsupported_to_supported_and_supported_to_none_are_known_policy_changes() {
        let (mut unsupported_lease, prepare_calls, _, policy_calls, _, _) =
            resident_admission_lease_with_policy(
                phase(),
                ResidentExecuteMode::Completed,
                None,
                Some(1),
                Some(policy()),
            );
        let returned =
            match unsupported_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                GpuBabBoundResidentWavePreparation::CleanDecline { reason, request } => {
                    assert_eq!(reason, GpuBabBoundResidentWaveDecline::Unsupported);
                    request
                }
                _ => panic!("initial None policy must clean-decline Unsupported"),
            };
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert!(matches!(
            unsupported_lease.prepare_resident_wave(returned),
            GpuBabBoundResidentWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendResidentAuthorityLost
            )
        ));
        assert_eq!(policy_calls.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(
            unsupported_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );

        let (mut none_lease, prepare_calls, _, policy_calls, _, _) =
            resident_admission_lease_with_policy(
                phase(),
                ResidentExecuteMode::Completed,
                Some(policy()),
                Some(1),
                None,
            );
        assert!(matches!(
            none_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())),
            GpuBabBoundResidentWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendResidentAuthorityLost
            )
        ));
        assert_eq!(policy_calls.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(
            none_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );

        let mut malformed_policy = policy();
        malformed_policy.maximum_slots = 0;
        let (mut malformed_lease, prepare_calls, _, policy_calls, _, _) =
            resident_admission_lease_with_policy(
                phase(),
                ResidentExecuteMode::Completed,
                Some(malformed_policy),
                None,
                None,
            );
        assert!(matches!(
            malformed_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())),
            GpuBabBoundResidentWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::InvalidResidentPolicy
            )
        ));
        assert_eq!(policy_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(
            malformed_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
    }

    #[test]
    fn v1_completed_restores_healthy_but_v1_failure_cannot_attest_refresh_only_resources() {
        let (mut completed_lease, completed_calls) =
            v1_refresh_only_lease(V1ExecuteMode::Completed);
        let capability = match completed_lease.prepare_wave(base_wave()) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("v1 may execute while only zero-byte RefreshOnly slots coexist"),
        };
        assert!(matches!(
            capability.execute_accepted(),
            GpuBabBoundWaveDisposition::Completed(_)
        ));
        assert_eq!(completed_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(completed_lease.resident_domains.live_counts(), (0, 2));
        assert_eq!(completed_lease.state, LeaseState::Open);
        assert_eq!(
            completed_lease.resource_certainty,
            ResidentResourceCertainty::HealthyKnown
        );

        let (mut failed_lease, failed_calls) =
            v1_refresh_only_lease(V1ExecuteMode::AcceptedFailure);
        let capability = match failed_lease.prepare_wave(base_wave()) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("v1 may execute while only zero-byte RefreshOnly slots coexist"),
        };
        let failure = match capability.execute_accepted() {
            GpuBabBoundWaveDisposition::AcceptedFailure(failure) => failure,
            _ => panic!("injected v1 accepted failure must remain terminal"),
        };
        assert_eq!(failed_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(failure.receipt_validated());
        assert_eq!(failed_lease.resident_domains.live_counts(), (0, 2));
        assert_eq!(failed_lease.state, LeaseState::Poisoned);
        assert_eq!(
            failed_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        match failed_lease.close() {
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                receipt_validated, ..
            } => assert!(!receipt_validated),
            GpuBabBoundPhaseCloseDisposition::Closed(_) => {
                panic!("v1 failure cannot authorize retained-v2 cleanup")
            }
        }
    }

    #[test]
    fn accepted_resident_failure_rolls_back_known_but_bad_receipt_stays_reserved_unknown() {
        let exact_phase = phase();
        let (mut exact_lease, _, exact_execute_calls) =
            resident_admission_lease_with_mode(exact_phase, ResidentExecuteMode::AcceptedFailure);
        let exact_failure =
            match exact_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                GpuBabBoundResidentWavePreparation::Accepted(capability) => {
                    match capability.execute_accepted() {
                        GpuBabBoundResidentWaveDisposition::AcceptedFailure(failure) => failure,
                        _ => panic!("exact raw failure must return an accepted failure"),
                    }
                }
                _ => panic!("fresh resident wave should be accepted"),
            };
        assert_eq!(exact_execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(
            exact_failure.receipt_validated(),
            "{}",
            exact_failure.detail()
        );
        assert!(exact_failure.host_audit().is_some());
        assert_eq!(
            exact_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
        assert!(exact_lease.resident_domains.resources_are_quiescent());
        assert!(exact_lease
            .resident_domains
            .slots
            .iter()
            .take(2)
            .all(|slot| matches!(
                slot,
                GpuBabBoundResidentSlotState::Vacant { high_generation: 1 }
            )));
        assert!(exact_lease
            .resident_domains
            .slots
            .iter()
            .skip(2)
            .all(|slot| matches!(
                slot,
                GpuBabBoundResidentSlotState::Vacant { high_generation: 0 }
            )));

        let phase = phase();
        let (mut malformed_lease, _, malformed_execute_calls) =
            resident_admission_lease_with_mode(phase, ResidentExecuteMode::MalformedFailure);
        let malformed_failure =
            match malformed_lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
                GpuBabBoundResidentWavePreparation::Accepted(capability) => {
                    match capability.execute_accepted() {
                        GpuBabBoundResidentWaveDisposition::AcceptedFailure(failure) => failure,
                        _ => panic!("malformed raw failure must be quarantined"),
                    }
                }
                _ => panic!("fresh resident wave should be accepted"),
            };
        assert_eq!(malformed_execute_calls.load(AtomicOrdering::Relaxed), 1);
        assert!(!malformed_failure.receipt_validated());
        assert!(malformed_failure.host_audit().is_some());
        assert_eq!(
            malformed_lease.resource_certainty,
            ResidentResourceCertainty::PoisonedUnknown
        );
        assert!(!malformed_lease.resident_domains.resources_are_quiescent());
        assert_eq!(malformed_lease.resident_domains.reserved_slots, 2);
        assert!(malformed_lease
            .resident_domains
            .slots
            .iter()
            .take(2)
            .all(|slot| matches!(slot, GpuBabBoundResidentSlotState::Reserved { .. })));
    }

    #[test]
    fn dropping_accepted_resident_wave_rolls_back_once_and_poison_is_known() {
        let phase = phase();
        let (mut lease, prepare_calls, execute_calls) = resident_admission_lease(phase);
        let capability = match lease.prepare_resident_wave(fresh_request_with_prefix(Vec::new())) {
            GpuBabBoundResidentWavePreparation::Accepted(capability) => capability,
            _ => panic!("fresh resident wave should be accepted"),
        };
        drop(capability);
        assert_eq!(prepare_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(execute_calls.load(AtomicOrdering::Relaxed), 0);
        let failure = lease
            .abandoned_resident_terminal()
            .expect("capability Drop records one sealed resident terminal");
        assert_eq!(
            failure.kind(),
            GpuBabBoundTerminalFailureKind::CapabilityAbandoned
        );
        assert!(failure.receipt_validated());
        assert_eq!(
            lease.resource_certainty,
            ResidentResourceCertainty::PoisonedKnown
        );
        assert!(lease.resident_domains.resources_are_quiescent());
        assert!(lease
            .resident_domains
            .slots
            .iter()
            .take(2)
            .all(|slot| matches!(
                slot,
                GpuBabBoundResidentSlotState::Vacant { high_generation: 1 }
            )));
    }

    #[test]
    fn append_children_accepts_strict_subset_without_cover_authority() {
        let first_inactive = literal(7, 3, GpuBabBoundSplitHistoryPhase::Inactive);
        let second_inactive = literal(8, 4, GpuBabBoundSplitHistoryPhase::Inactive);
        let first_active = literal(7, 3, GpuBabBoundSplitHistoryPhase::Active);
        let second_active = literal(8, 4, GpuBabBoundSplitHistoryPhase::Active);
        let mut words = Vec::new();
        words.extend(first_inactive);
        words.extend(second_inactive);
        words.extend(first_active);
        words.extend(second_active);
        let request = GpuBabBoundResidentWaveRequest::new(
            base_wave(),
            GpuBabBoundSplitHistoryArena::new(words),
            vec![GpuBabBoundResidentParentGroup {
                parent_group_id: 10,
                prefix: GpuBabBoundArenaRange { start: 0, len: 0 },
                construction: GpuBabBoundResidentConstruction::AppendReluChildren,
                source: GpuBabBoundResidentParentSource::FreshUpload { prior: None },
            }],
            vec![
                GpuBabBoundSplitHistoryView {
                    suffix: GpuBabBoundArenaRange { start: 0, len: 8 },
                    branch_pattern: 0,
                },
                GpuBabBoundSplitHistoryView {
                    suffix: GpuBabBoundArenaRange { start: 8, len: 8 },
                    branch_pattern: 3,
                },
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(request.wave.domains.len(), 2);
        assert_eq!(request.wave.domains[0].child_cardinality, 2);
        assert!(request.wave.domains[0].child_cardinality < (1 << 2));
        assert!(validate_history_for_test(&request, false).is_ok());
        let pending = planned(&request);
        assert_eq!(pending.destinations.len(), 2);
        assert_ne!(pending.schedule_identity_sha256, [0; 32]);
    }
}
