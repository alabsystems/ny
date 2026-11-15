// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Compact untrusted wire evidence; original-head replay retains all authority.
//!
//! Ranges name original token IDs, in exactly the ascending order required by
//! v1 evidence. They never name a permutation, cached geometry or a proof. The
//! succinct input is shape-bound and its full O(vocabulary) expansion is leased
//! before allocation, then sent to the unchanged arithmetic/coverage replay.

use super::certify::replay_with_owner;
use super::vocab_tree::{default_budget, validate_box, validate_head};
use super::{
    AffineHead, DecoderRefusal, GroupBounds, GroupEvidence, HiddenBox, Meter, NextTokenReport,
    SharedAffineHead, VocabEvidence, VocabOptions,
};
use crate::certify::work::{LeaseClass, WorkBudget, WorkLease, WorkRequest};
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use std::cell::{Cell, RefCell};
use std::io::Write;
use std::time::Instant;

#[cfg(test)]
thread_local! {
    static EXPORT_STOP_AT_MEMBER: Cell<Option<usize>> = const { Cell::new(None) };
}

#[cfg(test)]
pub(super) fn with_export_stop_after_members<T>(limit: usize, body: impl FnOnce() -> T) -> T {
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            EXPORT_STOP_AT_MEMBER.with(|stop| stop.set(self.0));
        }
    }
    let _restore = Restore(EXPORT_STOP_AT_MEMBER.with(|stop| stop.replace(Some(limit))));
    body()
}

/// The internal/direct-serde v1 form is deliberately unchanged. This borrowed
/// adapter emits maximal half-open runs without duplicating the member vectors.
pub(super) struct CompactEvidence<'a>(pub(super) &'a VocabEvidence);

/// Observed serialization work, separate from arithmetic/query visits and
/// conservative WorkBudget reservations. Encoded bytes are the actual output.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct WireExportStats {
    pub member_visits: usize,
    pub ranges_emitted: usize,
    pub groups_emitted: usize,
    pub encoded_bytes: usize,
    pub leased_scratch_bytes: usize,
}

/// The output allocation stays charged across independent replay. Drop this
/// owner only after its bytes are no longer needed; it conveys no proof.
#[derive(Debug)]
pub struct BudgetedWireExport {
    bytes: Vec<u8>,
    stats: WireExportStats,
    _lease: WorkLease,
}
impl BudgetedWireExport {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn stats(&self) -> &WireExportStats {
        &self.stats
    }
}

struct ExportMeter<'a> {
    budget: &'a WorkBudget,
    deadline: Option<Instant>,
    member_visits: Cell<usize>,
    ranges: Cell<usize>,
    groups: Cell<usize>,
    refusal: RefCell<Option<DecoderRefusal>>,
}
impl ExportMeter<'_> {
    fn checkpoint(&self) -> Result<(), DecoderRefusal> {
        #[cfg(test)]
        if EXPORT_STOP_AT_MEMBER.with(|stop| {
            stop.get()
                .is_some_and(|limit| self.member_visits.get() >= limit)
        }) {
            return Err(DecoderRefusal::Deadline);
        }
        self.budget
            .checkpoint_for(LeaseClass::Replay)
            .map_err(DecoderRefusal::Budget)?;
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(DecoderRefusal::Deadline);
        }
        Ok(())
    }
    fn observe(&self, count: &Cell<usize>) -> Result<(), DecoderRefusal> {
        let next = count
            .get()
            .checked_add(1)
            .ok_or(DecoderRefusal::Resource("export work overflow"))?;
        count.set(next);
        if next & 4095 == 0 {
            self.checkpoint()?;
        }
        Ok(())
    }
    fn serde_observe<E: serde::ser::Error>(&self, count: &Cell<usize>) -> Result<(), E> {
        self.observe(count).map_err(|refusal| {
            *self.refusal.borrow_mut() = Some(refusal);
            E::custom("export budget or deadline")
        })
    }
}

struct Ranges<'a> {
    members: &'a [usize],
    meter: Option<&'a ExportMeter<'a>>,
}
impl Serialize for Ranges<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(None)?;
        let mut run: Option<(usize, usize)> = None;
        for &id in self.members {
            if let Some(meter) = self.meter {
                meter.serde_observe::<S::Error>(&meter.member_visits)?;
            }
            let next_end = id
                .checked_add(1)
                .ok_or_else(|| serde::ser::Error::custom("range endpoint overflow"))?;
            match run {
                None => run = Some((id, next_end)),
                Some((start, end)) if id == end => run = Some((start, next_end)),
                Some((_, end)) if id < end => {
                    return Err(serde::ser::Error::custom("members are not ascending"));
                }
                Some((start, end)) => {
                    if let Some(meter) = self.meter {
                        meter.serde_observe::<S::Error>(&meter.ranges)?;
                    }
                    seq.serialize_element(&[start, end])?;
                    run = Some((id, next_end));
                }
            }
        }
        if let Some((start, end)) = run {
            if let Some(meter) = self.meter {
                meter.serde_observe::<S::Error>(&meter.ranges)?;
            }
            seq.serialize_element(&[start, end])?;
        }
        seq.end()
    }
}

struct CompactGroup<'a> {
    group: &'a GroupEvidence,
    meter: Option<&'a ExportMeter<'a>>,
}
impl Serialize for CompactGroup<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut out = serializer.serialize_struct("GroupEvidenceV2", 4)?;
        out.serialize_field(
            "member_ranges",
            &Ranges {
                members: &self.group.members,
                meter: self.meter,
            },
        )?;
        out.serialize_field("anchor_token", &self.group.anchor_token)?;
        out.serialize_field("group_bounds", &self.group.group_bounds)?;
        out.serialize_field("upper", &self.group.upper)?;
        out.end()
    }
}

struct CompactGroups<'a> {
    groups: &'a [GroupEvidence],
    meter: Option<&'a ExportMeter<'a>>,
}
impl Serialize for CompactGroups<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut out = serializer.serialize_seq(Some(self.groups.len()))?;
        for group in self.groups {
            if let Some(meter) = self.meter {
                meter.serde_observe::<S::Error>(&meter.groups)?;
            }
            out.serialize_element(&CompactGroup {
                group,
                meter: self.meter,
            })?;
        }
        out.end()
    }
}

struct MeteredEvidence<'a> {
    evidence: &'a VocabEvidence,
    meter: Option<&'a ExportMeter<'a>>,
}

impl Serialize for MeteredEvidence<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.evidence.schema_version != 1 {
            return Err(serde::ser::Error::custom("unsupported internal schema"));
        }
        let mut out = serializer.serialize_struct("VocabEvidenceV2", 6)?;
        out.serialize_field("schema_version", &2u32)?;
        out.serialize_field("scope_digest", &self.evidence.scope_digest)?;
        out.serialize_field("target", &self.evidence.target)?;
        out.serialize_field("vocab_size", &self.evidence.vocab_size)?;
        out.serialize_field("hidden_size", &self.evidence.hidden_size)?;
        out.serialize_field(
            "groups",
            &CompactGroups {
                groups: &self.evidence.groups,
                meter: self.meter,
            },
        )?;
        out.end()
    }
}

impl Serialize for CompactEvidence<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MeteredEvidence {
            evidence: self.0,
            meter: None,
        }
        .serialize(serializer)
    }
}

struct BoundedOutput {
    bytes: Vec<u8>,
    cap: usize,
    refused: bool,
}
impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|len| len > self.cap)
        {
            self.refused = true;
            return Err(std::io::Error::other("JSON export byte cap"));
        }
        // The admitted exact-capacity buffer is allocated before serialization;
        // writes do not grow it and cannot bypass the byte limit.
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn export_with_budget(
    evidence: &VocabEvidence,
    options: &VocabOptions,
) -> Result<BudgetedWireExport, DecoderRefusal> {
    let budget = options
        .budget
        .as_ref()
        .ok_or(DecoderRefusal::Resource("export requires shared ledger"))?;
    let meter = ExportMeter {
        budget,
        deadline: options.deadline,
        member_visits: Cell::new(0),
        ranges: Cell::new(0),
        groups: Cell::new(0),
        refusal: RefCell::new(None),
    };
    meter.checkpoint()?;
    if evidence.schema_version != 1
        || evidence.vocab_size == 0
        || evidence.vocab_size > options.max_vocab
        || evidence.hidden_size == 0
        || evidence.hidden_size > options.max_hidden
        || evidence.target >= evidence.vocab_size
        || evidence.groups.len() > options.max_nodes
    {
        return Err(DecoderRefusal::Resource("export shape cap"));
    }
    let mut member_count = 0usize;
    for (index, group) in evidence.groups.iter().enumerate() {
        if index & 4095 == 0 {
            meter.checkpoint()?;
        }
        member_count = add(member_count, group.members.len())?;
    }
    if member_count != evidence.vocab_size - 1 {
        return Err(DecoderRefusal::Evidence("export member count"));
    }
    let cap = options
        .max_evidence_bytes
        .min(usize::try_from(options.max_scratch_bytes / 16).unwrap_or(usize::MAX));
    let scratch = add(cap, size_of::<BudgetedWireExport>())?;
    let work = add(
        add(mul(member_count, 2)?, mul(evidence.groups.len(), 32)?)?,
        cap,
    )? as u64;
    if scratch as u64 > options.max_scratch_bytes || work > options.max_terms {
        return Err(DecoderRefusal::Resource("export memory or work cap"));
    }
    let mut lease = budget
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                scratch_bytes: scratch,
                persistent_bytes: 0,
                work,
                proposals: 1,
            },
        )
        .map_err(DecoderRefusal::Budget)?;
    let mut output = BoundedOutput {
        bytes: Vec::new(),
        cap,
        refused: false,
    };
    output
        .bytes
        .try_reserve_exact(cap)
        .map_err(|_| DecoderRefusal::Resource("export allocation"))?;
    let encoded = serde_json::to_writer(
        &mut output,
        &MeteredEvidence {
            evidence,
            meter: Some(&meter),
        },
    );
    if encoded.is_err() {
        if let Some(refusal) = meter.refusal.take() {
            return Err(refusal);
        }
        return Err(if output.refused {
            DecoderRefusal::Resource("JSON export byte cap")
        } else {
            DecoderRefusal::Evidence("serialization")
        });
    }
    meter.checkpoint()?;
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let stats = WireExportStats {
        member_visits: meter.member_visits.get(),
        ranges_emitted: meter.ranges.get(),
        groups_emitted: meter.groups.get(),
        encoded_bytes: output.bytes.len(),
        leased_scratch_bytes: scratch,
    };
    Ok(BudgetedWireExport {
        bytes: output.bytes,
        stats,
        _lease: lease,
    })
}

pub(super) fn write_compact_json(
    evidence: &VocabEvidence,
    writer: impl Write,
) -> Result<(), DecoderRefusal> {
    serde_json::to_writer(writer, &CompactEvidence(evidence))
        .map_err(|_| DecoderRefusal::Evidence("serialization"))
}

/// This first pass ignores payloads without allocating ranges or member lists.
/// Serde rejects duplicate required fields and enforces its recursion limit.
#[derive(Deserialize)]
struct Header {
    schema_version: u32,
    target: usize,
    vocab_size: usize,
    hidden_size: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RangeGroup {
    member_ranges: Vec<[usize; 2]>,
    anchor_token: usize,
    group_bounds: Option<GroupBounds>,
    upper: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RangeEvidence {
    schema_version: u32,
    scope_digest: [u8; 32],
    target: usize,
    vocab_size: usize,
    hidden_size: usize,
    groups: Vec<RangeGroup>,
}

fn add(a: usize, b: usize) -> Result<usize, DecoderRefusal> {
    a.checked_add(b)
        .ok_or(DecoderRefusal::Resource("range expansion size"))
}
fn mul(a: usize, b: usize) -> Result<usize, DecoderRefusal> {
    a.checked_mul(b)
        .ok_or(DecoderRefusal::Resource("range expansion size"))
}

fn expand(
    evidence: &RangeEvidence,
    options: &VocabOptions,
    meter: &mut Meter<'_>,
) -> Result<(VocabEvidence, WorkLease), DecoderRefusal> {
    // Canonicality and total size are checked before bitmap/member allocation.
    if evidence.schema_version != 2 || evidence.groups.len() > options.max_nodes {
        return Err(DecoderRefusal::Evidence("range schema or node cap"));
    }
    let mut total = 0usize;
    for group in &evidence.groups {
        meter.charge(1)?;
        if group.member_ranges.is_empty() || !group.upper.is_finite() {
            return Err(DecoderRefusal::Evidence("range group shape or bound"));
        }
        let mut previous_end = None;
        let mut has_anchor = false;
        for &[start, end] in &group.member_ranges {
            meter.charge(1)?;
            if start >= end
                || end > evidence.vocab_size
                || previous_end.is_some_and(|previous| start <= previous)
                || (start <= evidence.target && evidence.target < end)
            {
                return Err(DecoderRefusal::Evidence("invalid or noncanonical range"));
            }
            total = add(total, end - start)?;
            has_anchor |= start <= group.anchor_token && group.anchor_token < end;
            previous_end = Some(end);
        }
        if !has_anchor {
            return Err(DecoderRefusal::Evidence("range anchor is not a member"));
        }
    }
    if total != evidence.vocab_size - 1 {
        return Err(DecoderRefusal::Evidence("incomplete range vocabulary"));
    }
    let bytes = add(
        add(mul(total, size_of::<usize>())?, evidence.vocab_size)?,
        add(
            mul(evidence.groups.len(), size_of::<GroupEvidence>())?,
            size_of::<VocabEvidence>(),
        )?,
    )?;
    if bytes as u64 > options.max_scratch_bytes {
        return Err(DecoderRefusal::Resource("range expansion memory cap"));
    }
    // The caller keeps this reservation alive through original-head replay.
    // Construction resides in scratch; replay separately prices its input,
    // output, original head and arithmetic scratch. No fixed encoded-byte
    // multiplier substitutes for this explicit expansion charge.
    let budget = options
        .budget
        .as_ref()
        .ok_or(DecoderRefusal::Resource("range expansion ledger"))?;
    let work = add(mul(evidence.vocab_size, 4)?, total)? as u64;
    let lease = budget
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                scratch_bytes: bytes,
                persistent_bytes: 0,
                work,
                proposals: 1,
            },
        )
        .map_err(DecoderRefusal::Budget)?;
    let mut seen = Vec::<u8>::new();
    seen.try_reserve_exact(evidence.vocab_size)
        .map_err(|_| DecoderRefusal::Resource("range coverage allocation"))?;
    seen.resize(evidence.vocab_size, 0);
    for group in &evidence.groups {
        for &[start, end] in &group.member_ranges {
            for offset in (start..end).step_by(4096) {
                let stop = offset.saturating_add(4096).min(end);
                meter.charge(stop - offset)?;
                for value in &mut seen[offset..stop] {
                    if *value != 0 {
                        return Err(DecoderRefusal::Evidence("overlapping range groups"));
                    }
                    *value = 1;
                }
            }
        }
    }
    for (index, values) in seen.chunks(4096).enumerate() {
        meter.charge(values.len())?;
        if values
            .iter()
            .enumerate()
            .any(|(offset, &value)| index * 4096 + offset != evidence.target && value != 1)
        {
            return Err(DecoderRefusal::Evidence("range vocabulary hole"));
        }
    }
    drop(seen);
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(evidence.groups.len())
        .map_err(|_| DecoderRefusal::Resource("range group allocation"))?;
    for group in &evidence.groups {
        let count = group
            .member_ranges
            .iter()
            .try_fold(0usize, |count, &[start, end]| {
                count.checked_add(end - start)
            })
            .ok_or(DecoderRefusal::Resource("range member size"))?;
        let mut members = Vec::new();
        members
            .try_reserve_exact(count)
            .map_err(|_| DecoderRefusal::Resource("range member allocation"))?;
        for &[start, end] in &group.member_ranges {
            for offset in (start..end).step_by(4096) {
                let stop = offset.saturating_add(4096).min(end);
                meter.charge(stop - offset)?;
                members.extend(offset..stop);
            }
        }
        groups.push(GroupEvidence {
            members,
            anchor_token: group.anchor_token,
            group_bounds: group.group_bounds.clone(),
            upper: group.upper,
        });
    }
    // A typed allocation owner carries the lease across the independent replay.
    // It is returned by the helper below rather than releasing bytes early.
    let expanded = VocabEvidence {
        schema_version: 1,
        scope_digest: evidence.scope_digest,
        target: evidence.target,
        vocab_size: evidence.vocab_size,
        hidden_size: evidence.hidden_size,
        groups,
    };
    Ok((expanded, lease))
}

/// Byte admission precedes both parses. v2 gets a separate checked expansion
/// lease; v1 retains its existing explicit-ID representation and admission.
pub(super) fn replay_json(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    json: &[u8],
    options: &VocabOptions,
) -> Result<NextTokenReport, DecoderRefusal> {
    replay_json_with_owner(head, hidden, anchor, target, json, options, None)
}

pub(super) fn replay_json_with_owner(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    json: &[u8],
    options: &VocabOptions,
    head_owner: Option<SharedAffineHead>,
) -> Result<NextTokenReport, DecoderRefusal> {
    let scoped = head_owner
        .as_ref()
        .map(|owner| owner.scoped_options(options))
        .transpose()?;
    let options = scoped.as_ref().unwrap_or(options);
    if let Some(owner) = &head_owner {
        owner.validate_binding(head)?;
    }
    validate_head(head, anchor, target, options)?;
    validate_box(hidden, head.hidden_size)?;
    super::execution::validate_input_domain(head.arithmetic, hidden)?;
    let domain_work = if matches!(head.arithmetic, super::HeadArithmetic::Binary32 { .. }) {
        head.hidden_size
    } else {
        0
    };
    let parsed_bytes = mul(json.len(), 16)?;
    if json.len() > options.max_evidence_bytes || parsed_bytes as u64 > options.max_scratch_bytes {
        return Err(DecoderRefusal::Resource("JSON admission"));
    }
    let budget = default_budget(options)?;
    let mut meter = Meter::with_budget_class(options, budget.clone(), LeaseClass::Replay)?;
    meter.charge(domain_work)?;
    let mut parsing = budget
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                scratch_bytes: parsed_bytes,
                persistent_bytes: 0,
                work: add(mul(json.len(), 2)?, domain_work)? as u64,
                proposals: 1,
            },
        )
        .map_err(DecoderRefusal::Budget)?;
    let header: Header =
        serde_json::from_slice(json).map_err(|_| DecoderRefusal::Evidence("JSON header"))?;
    meter.charge(json.len())?;
    if header.vocab_size != head.vocab_size
        || header.hidden_size != head.hidden_size
        || header.target != target
        || !matches!(header.schema_version, 1 | 2)
    {
        return Err(DecoderRefusal::Evidence("JSON request or schema binding"));
    }
    let mut replay_options = options.clone();
    replay_options.budget = Some(budget);
    let report = match header.schema_version {
        1 => {
            let evidence: VocabEvidence = serde_json::from_slice(json)
                .map_err(|_| DecoderRefusal::Evidence("JSON schema"))?;
            meter.charge(json.len())?;
            replay_with_owner(
                head,
                hidden,
                anchor,
                target,
                &evidence,
                &replay_options,
                head_owner,
            )?
        }
        2 => {
            let evidence: RangeEvidence = serde_json::from_slice(json)
                .map_err(|_| DecoderRefusal::Evidence("JSON range schema"))?;
            meter.charge(json.len())?;
            let (expanded, mut expansion) = expand(&evidence, &replay_options, &mut meter)?;
            let report = replay_with_owner(
                head,
                hidden,
                anchor,
                target,
                &expanded,
                &replay_options,
                head_owner,
            )?;
            drop(expanded);
            expansion.complete().map_err(DecoderRefusal::Budget)?;
            report
        }
        _ => unreachable!("header schema was validated"),
    };
    parsing.complete().map_err(DecoderRefusal::Budget)?;
    Ok(report)
}
