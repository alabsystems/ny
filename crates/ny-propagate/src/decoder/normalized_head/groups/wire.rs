// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded private schema2 wire for canonical original Group/Row evidence.
//!
//! Only original-owner replay conveys authority. Both streaming passes check
//! the complete deterministic cover. The first pass allocates no entry vector;
//! original full-head/norm/box scope hashing precedes the second pass. This
//! closed format requires the field order emitted by this module and raw serde
//! for these exact evidence types. No cache geometry or method tag is trusted.
//! A group denotes its original half-open range with the bound target excluded,
//! exactly as in the existing private replay, including a target inside a range.

use super::*;
use std::io::{self, Write};

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct GroupedWireStats {
    /// Bytes advanced by the two streaming cursors, excluding bounded scalar
    /// token rereads; their worst-case work is prepaid separately below.
    pub(crate) input_byte_visits: usize,
    pub(crate) original_identity_visits: u64,
    pub(crate) parsed_entry_visits: usize,
    pub(crate) coverage_visits: usize,
    pub(crate) validated_entries: usize,
    pub(crate) exported_entries: usize,
    pub(crate) encoded_bytes: usize,
    /// This wire phase only; sealed arithmetic replay has its own spent phases.
    pub(crate) reserved_work: u64,
}

#[derive(Debug)]
pub(crate) struct GroupedWireExport {
    bytes: Vec<u8>,
    stats: GroupedWireStats,
    reservation: PersistentReservation,
}
impl GroupedWireExport {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub(crate) fn stats(&self) -> &GroupedWireStats {
        &self.stats
    }
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
}

#[derive(Debug)]
pub(crate) struct GroupedWireReplay {
    pub(crate) report: NormalizedGroupedHeadReport,
    pub(crate) stats: GroupedWireStats,
}

pub(crate) fn grouped_wire_size_bound(vocab: usize) -> Result<usize> {
    if vocab == 0 {
        return Err(DecoderRefusal::Shape("grouped wire vocabulary").into());
    }
    // Every terminal uses <=192 bytes (fixed keys/tags, at most two usize
    // endpoints and one finite Ryu f64). The header, digest and width fit2048.
    add(mul(vocab - 1, 192)?, 2048)
}

fn export_work(cap: usize, vocab: usize) -> Result<u64> {
    u64::try_from(add(cap, add(mul(vocab, 2)?, 8192)?)?)
        .map_err(|_| resource("grouped wire export work overflow"))
}

fn decode_request(bytes: usize, vocab: usize, hidden: usize) -> Result<WorkRequest> {
    if vocab == 0 || hidden == 0 {
        return Err(DecoderRefusal::Shape("grouped wire original dimensions").into());
    }
    // Reserve maximum original entries with allocator slack BEFORE the first
    // scan. No compact group can hide an O(V) expansion in its encoded size.
    let persistent = add(
        mul(mul(vocab - 1, size_of::<NormalizedTerminalEvidence>())?, 2)?,
        4096,
    )?;
    // Logical borrowed input bytes plus possible scope-copy capacity. A caller
    // owned export remains separately charged; this is a conservative overlap.
    let scratch = add(bytes, add(mul(hidden, 32)?, 16 * 1024)?)?;
    let originals = add(mul(vocab, hidden)?, vocab)?;
    // Each pass advances once and can reread an integer token for digit,
    // UTF-8 and bounded scalar parsing. Eight visits/byte cover both passes
    // without treating the cursor counter as all lexical inspection work.
    let work = add(
        mul(bytes, 8)?,
        add(
            mul(originals, 2)?,
            add(mul(hidden, 32)?, add(mul(vocab, 4)?, 8192)?)?,
        )?,
    )?;
    Ok(WorkRequest {
        persistent_bytes: persistent,
        scratch_bytes: scratch,
        work: u64::try_from(work).map_err(|_| resource("grouped wire decode work overflow"))?,
        proposals: 1,
    })
}

/// Publication census on ONE caller ledger: export, two-pass bounded decode
/// including full original identity, then worst old/member/direct replay. This
/// predicts work, never reserves/refills an allowance or reports runtime/RSS.
pub(crate) fn grouped_publication_work_bound(
    vocab: usize,
    hidden: usize,
    width: usize,
) -> Result<u64> {
    let cap = grouped_wire_size_bound(vocab)?;
    checked_work_add(
        export_work(cap, vocab)?,
        checked_work_add(
            decode_request(cap, vocab, hidden)?.work,
            normalized_group_work_bound(vocab, hidden, width)?,
        )?,
    )
}

fn failed() -> NormalizedHeadRefusal {
    evidence_error("grouped wire header/format/complete cover")
}

fn ledger_checkpoint(ledger: &WorkBudget, limits: &VocabOptions) -> Result<()> {
    ledger
        .checkpoint_for(LeaseClass::Replay)
        .map_err(DecoderRefusal::Budget)?;
    if limits.deadline.is_some_and(|d| Instant::now() >= d)
        || ny_core::instance_budget::deadline().is_some_and(|d| Instant::now() >= d)
    {
        return Err(DecoderRefusal::Deadline.into());
    }
    Ok(())
}

fn live_capacity(
    ledger: &WorkBudget,
    owner: &SharedAffineHead,
    request: WorkRequest,
    limits: &VocabOptions,
) -> Result<()> {
    let live = ledger.receipt().map_err(DecoderRefusal::Budget)?;
    if add(
        add(
            live.live_persistent_bytes,
            owner.resident_bytes_outside_ledger(ledger),
        )?,
        request.persistent_bytes,
    )? as u64
        > limits.max_persistent_bytes
        || add(live.live_scratch_bytes, request.scratch_bytes)? as u64 > limits.max_scratch_bytes
        || request.work > limits.max_terms
        || request.proposals > limits.max_proposals
    {
        return Err(resource("grouped aggregate wire capacity/work"));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Cover,
    Materialize,
}

struct Parser<'a, 'b> {
    bytes: &'a [u8],
    pos: usize,
    lease: &'b WorkLease,
    limits: &'b VocabOptions,
    pass: Pass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Header {
    digest: [u8; 32],
    width: usize,
    entries: usize,
}

impl<'a> Parser<'a, '_> {
    fn advance(&mut self) -> Result<()> {
        self.pos += 1;
        #[cfg(test)]
        inspect_byte(self.pass, self.pos);
        if self.pos.is_multiple_of(1024) {
            checkpoint(self.lease, self.limits)?;
        }
        Ok(())
    }
    fn ws(&mut self) -> Result<()> {
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| matches!(*b, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.advance()?;
        }
        Ok(())
    }
    fn expect(&mut self, byte: u8) -> Result<()> {
        self.ws()?;
        if self.bytes.get(self.pos) != Some(&byte) {
            return Err(failed());
        }
        self.advance()
    }
    fn string(&mut self) -> Result<&'a str> {
        self.expect(b'"')?;
        let start = self.pos;
        loop {
            let byte = *self.bytes.get(self.pos).ok_or_else(failed)?;
            if byte == b'"' {
                break;
            }
            if !(32..=126).contains(&byte) || byte == b'\\' || self.pos - start >= 40 {
                return Err(failed());
            }
            self.advance()?;
        }
        let end = self.pos;
        self.advance()?;
        std::str::from_utf8(&self.bytes[start..end]).map_err(|_| failed())
    }
    fn key(&mut self, expected: &str) -> Result<()> {
        if self.string()? != expected {
            return Err(failed());
        }
        self.expect(b':')
    }
    fn next_key(&mut self, expected: &str) -> Result<()> {
        self.expect(b',')?;
        self.key(expected)
    }
    fn number(&mut self) -> Result<&'a [u8]> {
        self.ws()?;
        let start = self.pos;
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| b.is_ascii_digit() || matches!(*b, b'-' | b'+' | b'.' | b'e' | b'E'))
        {
            if self.pos - start >= 64 {
                return Err(failed());
            }
            self.advance()?;
        }
        if start == self.pos {
            return Err(failed());
        }
        Ok(&self.bytes[start..self.pos])
    }
    fn uint(&mut self) -> Result<usize> {
        let token = self.number()?;
        if token.iter().any(|b| !b.is_ascii_digit()) || (token.len() > 1 && token[0] == b'0') {
            return Err(failed());
        }
        std::str::from_utf8(token)
            .map_err(|_| failed())?
            .parse()
            .map_err(|_| failed())
    }
    fn float(&mut self) -> Result<f64> {
        let value: f64 = serde_json::from_slice(self.number()?).map_err(|_| failed())?;
        if !value.is_finite() {
            return Err(failed());
        }
        Ok(value)
    }
    fn digest(&mut self) -> Result<[u8; 32]> {
        self.expect(b'[')?;
        let mut digest = [0; 32];
        for (i, byte) in digest.iter_mut().enumerate() {
            if i != 0 {
                self.expect(b',')?;
            }
            *byte = u8::try_from(self.uint()?).map_err(|_| failed())?;
        }
        self.expect(b']')?;
        Ok(digest)
    }
    fn entry(&mut self) -> Result<NormalizedTerminalEvidence> {
        self.expect(b'{')?;
        self.key("kind")?;
        let result = match self.string()? {
            "OriginalGroup" => {
                self.next_key("start")?;
                let start = self.uint()?;
                self.next_key("end")?;
                let end = self.uint()?;
                self.next_key("upper")?;
                let upper = self.float()?;
                NormalizedTerminalEvidence::OriginalGroup { start, end, upper }
            }
            "OriginalRow" => {
                self.next_key("competitor")?;
                let competitor = self.uint()?;
                self.next_key("upper")?;
                let upper = self.float()?;
                self.next_key("method")?;
                let method = match self.string()? {
                    "ProjectedDense" => NormalizedSupportMethod::ProjectedDense,
                    "RadialTaylor" => NormalizedSupportMethod::RadialTaylor,
                    _ => return Err(failed()),
                };
                NormalizedTerminalEvidence::OriginalRow {
                    competitor,
                    upper,
                    method,
                }
            }
            _ => return Err(failed()),
        };
        self.expect(b'}')?;
        Ok(result)
    }
    fn next_entry(
        &mut self,
        count: &mut usize,
        vocab: usize,
    ) -> Result<NormalizedTerminalEvidence> {
        if *count >= vocab - 1 {
            return Err(failed());
        }
        if *count != 0 {
            self.expect(b',')?;
        }
        checkpoint(self.lease, self.limits)?;
        let entry = self.entry()?;
        *count += 1;
        Ok(entry)
    }
    fn document(
        &mut self,
        vocab: usize,
        hidden: usize,
        target: usize,
        norm: FinalNormalizationKind,
        mut output: Option<&mut Vec<NormalizedTerminalEvidence>>,
    ) -> Result<Header> {
        checkpoint(self.lease, self.limits)?;
        self.expect(b'{')?;
        self.key("schema_version")?;
        if self.uint()? != GROUP_SCHEMA as usize {
            return Err(failed());
        }
        self.next_key("scope_digest")?;
        let digest = self.digest()?;
        self.next_key("target")?;
        if self.uint()? != target {
            return Err(failed());
        }
        self.next_key("vocab_size")?;
        if self.uint()? != vocab {
            return Err(failed());
        }
        self.next_key("hidden_size")?;
        if self.uint()? != hidden {
            return Err(failed());
        }
        self.next_key("normalization")?;
        let declared = match self.string()? {
            "LayerNorm" => FinalNormalizationKind::LayerNorm,
            "RmsNorm" => FinalNormalizationKind::RmsNorm,
            _ => return Err(failed()),
        };
        if declared != norm {
            return Err(failed());
        }
        self.next_key("group_width")?;
        let width = self.uint()?;
        if width == 0 {
            return Err(failed());
        }
        self.next_key("entries")?;
        self.expect(b'[')?;
        let mut count = 0;
        let mut covered = 0;
        for (start, end) in Ranges::new(vocab, target, width) {
            let first = self.next_entry(&mut count, vocab)?;
            match first {
                NormalizedTerminalEvidence::OriginalGroup {
                    start: s, end: e, ..
                } => {
                    if s != start || e != end {
                        return Err(failed());
                    }
                    for id in start..end {
                        if id == target {
                            continue;
                        }
                        if covered % CHUNK == 0 {
                            checkpoint(self.lease, self.limits)?;
                        }
                        covered += 1;
                    }
                    if let Some(entries) = &mut output {
                        entries.push(first);
                    }
                }
                NormalizedTerminalEvidence::OriginalRow { .. } => {
                    let mut first = Some(first);
                    for id in start..end {
                        if id == target {
                            continue;
                        }
                        let row = match first.take() {
                            Some(row) => row,
                            None => self.next_entry(&mut count, vocab)?,
                        };
                        if !matches!(row, NormalizedTerminalEvidence::OriginalRow { competitor, .. } if competitor == id)
                        {
                            return Err(failed());
                        }
                        if covered % CHUNK == 0 {
                            checkpoint(self.lease, self.limits)?;
                        }
                        covered += 1;
                        if let Some(entries) = &mut output {
                            entries.push(row);
                        }
                    }
                }
            }
        }
        self.expect(b']')?;
        self.expect(b'}')?;
        self.ws()?;
        if covered != vocab - 1 || self.pos != self.bytes.len() {
            return Err(failed());
        }
        checkpoint(self.lease, self.limits)?;
        Ok(Header {
            digest,
            width,
            entries: count,
        })
    }
}

pub(crate) fn replay_shared_grouped_normalized_json(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    bytes: &[u8],
    limits: &VocabOptions,
) -> Result<GroupedWireReplay> {
    let ledger = owner.budget();
    ledger_checkpoint(&ledger, limits)?;
    environment()?;
    let head = owner.as_head();
    let options = NormalizedHeadOptions {
        budget: &ledger,
        limits,
    };
    let _ = shape_with_owner(head, norm, pre, target, options, Some(owner))?;
    if bytes.len() > limits.max_evidence_bytes {
        return Err(resource("grouped wire input byte cap"));
    }
    if add(
        mul(head.vocab_size - 1, size_of::<NormalizedTerminalEvidence>())?,
        size_of::<NormalizedGroupedHeadEvidence>(),
    )? > limits.max_evidence_bytes
    {
        return Err(resource("grouped wire maximum decoded evidence cap"));
    }
    let request = decode_request(bytes.len(), head.vocab_size, head.hidden_size)?;
    live_capacity(&ledger, owner, request, limits)?;
    let mut lease = ledger
        .lease_for(LeaseClass::Replay, request)
        .map_err(DecoderRefusal::Budget)?;
    checkpoint(&lease, limits)?;
    let declared = Parser {
        bytes,
        pos: 0,
        lease: &lease,
        limits,
        pass: Pass::Cover,
    }
    .document(head.vocab_size, head.hidden_size, target, norm.kind, None)?;
    // Full original identity check is mandatory even when every proposal says
    // "one cached group". These temporary scope arrays are priced by scratch.
    lease.mark_materialized().map_err(DecoderRefusal::Budget)?;
    let mut identity_stats = NormalizedHeadStats::default();
    let (original_scope, actual_digest) = scope(
        head,
        norm,
        pre,
        target,
        &lease,
        limits,
        &mut identity_stats,
        Some(owner),
    )?;
    drop(original_scope);
    if declared.digest != actual_digest {
        return Err(evidence_error("grouped wire original scope binding"));
    }
    let mut entries = reserve(head.vocab_size - 1)?;
    if add(
        mul(entries.capacity(), size_of::<NormalizedTerminalEvidence>())?,
        4096,
    )? > request.persistent_bytes
    {
        return Err(resource("grouped wire actual entry capacity"));
    }
    let repeated = Parser {
        bytes,
        pos: 0,
        lease: &lease,
        limits,
        pass: Pass::Materialize,
    }
    .document(
        head.vocab_size,
        head.hidden_size,
        target,
        norm.kind,
        Some(&mut entries),
    )?;
    if repeated != declared {
        return Err(failed());
    }
    let evidence = NormalizedGroupedHeadEvidence {
        schema_version: GROUP_SCHEMA,
        scope_digest: actual_digest,
        target,
        vocab_size: head.vocab_size,
        hidden_size: head.hidden_size,
        normalization: norm.kind,
        group_width: declared.width,
        entries,
    };
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let expansion = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
    let report =
        replay_shared_grouped_normalized_next_token(owner, norm, pre, target, &evidence, options)?;
    drop(evidence);
    drop(expansion);
    ledger_checkpoint(&ledger, limits)?;
    Ok(GroupedWireReplay {
        report,
        stats: GroupedWireStats {
            input_byte_visits: mul(bytes.len(), 2)?,
            original_identity_visits: identity_stats.original_head_visits
                + identity_stats.normalization_parameter_visits
                + identity_stats.region_visits,
            parsed_entry_visits: mul(declared.entries, 2)?,
            coverage_visits: mul(head.vocab_size - 1, 2)?,
            encoded_bytes: bytes.len(),
            reserved_work: request.work,
            ..GroupedWireStats::default()
        },
    })
}

struct Writer<'a> {
    bytes: Vec<u8>,
    cap: usize,
    lease: &'a WorkLease,
    limits: &'a VocabOptions,
}
impl Write for Writer<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        checkpoint(self.lease, self.limits).map_err(io::Error::other)?;
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > self.cap)
        {
            return Err(io::Error::other("grouped wire encoded cap"));
        }
        for chunk in bytes.chunks(1024) {
            checkpoint(self.lease, self.limits).map_err(io::Error::other)?;
            self.bytes.extend_from_slice(chunk);
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn write_error(lease: &WorkLease, limits: &VocabOptions) -> NormalizedHeadRefusal {
    checkpoint(lease, limits)
        .err()
        .map_or_else(failed, Into::into)
}

/// Editable evidence remains an untrusted proposal, including after export.
/// Only the separate fresh original replay can mint a checked outcome.
pub(crate) fn export_shared_grouped_normalized_json(
    owner: &SharedAffineHead,
    evidence: &NormalizedGroupedHeadEvidence,
    limits: &VocabOptions,
) -> Result<GroupedWireExport> {
    let ledger = owner.budget();
    let _ = owner.scoped_options(limits)?;
    ledger_checkpoint(&ledger, limits)?;
    let head = owner.as_head();
    if head.arithmetic != HeadArithmetic::ExactReal {
        return Err(NormalizedHeadRefusal::UnsupportedHeadArithmetic(
            head.arithmetic,
        ));
    }
    // Export has no normalization/box authority: inspect only fixed original
    // head metadata and the editable proposal's complete canonical cover.
    if evidence.schema_version != GROUP_SCHEMA
        || evidence.vocab_size != head.vocab_size
        || evidence.hidden_size != head.hidden_size
        || evidence.target >= head.vocab_size
        || evidence.group_width == 0
        || evidence.entries.len() > head.vocab_size - 1
    {
        return Err(failed());
    }
    if add(
        mul(
            evidence.entries.capacity(),
            size_of::<NormalizedTerminalEvidence>(),
        )?,
        size_of::<NormalizedGroupedHeadEvidence>(),
    )? > limits.max_evidence_bytes
    {
        return Err(resource("grouped wire proposal capacity"));
    }
    let cap = grouped_wire_size_bound(head.vocab_size)?;
    if cap > limits.max_evidence_bytes {
        return Err(resource("grouped wire output byte cap"));
    }
    let request = WorkRequest {
        persistent_bytes: cap,
        scratch_bytes: 16 * 1024,
        work: export_work(cap, head.vocab_size)?,
        proposals: 1,
    };
    live_capacity(&ledger, owner, request, limits)?;
    let mut lease = ledger
        .lease_for(LeaseClass::Replay, request)
        .map_err(DecoderRefusal::Budget)?;
    checkpoint(&lease, limits)?;
    let mut covered = 0;
    coverage(evidence, &lease, limits, |_, _| {
        covered += 1;
        Ok(())
    })?;
    lease.mark_materialized().map_err(DecoderRefusal::Budget)?;
    let mut writer = Writer {
        bytes: reserve(cap)?,
        cap,
        lease: &lease,
        limits,
    };
    if writer.bytes.capacity() > cap {
        return Err(resource("grouped wire actual output capacity"));
    }
    serde_json::to_writer(&mut writer, evidence).map_err(|_| write_error(&lease, limits))?;
    let encoded = writer.bytes.len();
    let bytes = writer.bytes;
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let reservation = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
    ledger_checkpoint(&ledger, limits)?;
    Ok(GroupedWireExport {
        bytes,
        stats: GroupedWireStats {
            coverage_visits: covered,
            validated_entries: evidence.entries.len(),
            exported_entries: evidence.entries.len(),
            encoded_bytes: encoded,
            reserved_work: request.work,
            ..GroupedWireStats::default()
        },
        reservation,
    })
}

#[cfg(test)]
struct Stop {
    pass: Pass,
    after: usize,
    cancellation: crate::certify::residency::CancellationToken,
}
#[cfg(test)]
thread_local! { static STOP:std::cell::RefCell<Option<Stop>>=const { std::cell::RefCell::new(None) }; }
#[cfg(test)]
fn inspect_byte(pass: Pass, pos: usize) {
    STOP.with(|stop| {
        if let Some(stop) = stop.borrow().as_ref() {
            if stop.pass == pass && pos >= stop.after {
                stop.cancellation.cancel();
            }
        }
    });
}
#[cfg(test)]
fn cancel_after<T>(
    pass: Pass,
    after: usize,
    cancellation: crate::certify::residency::CancellationToken,
    body: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<Stop>);
    impl Drop for Restore {
        fn drop(&mut self) {
            STOP.with(|stop| {
                stop.replace(self.0.take());
            });
        }
    }
    let _restore = Restore(STOP.with(|stop| {
        stop.replace(Some(Stop {
            pass,
            after,
            cancellation,
        }))
    }));
    body()
}

#[cfg(test)]
mod tests;
