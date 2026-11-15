// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Distinct bounded normalized-head wire. It has no arithmetic authority.
//!
//! Two streaming passes bind the complete header/coverage before allocating
//! rows. Original head/norm/box hashing precedes expansion. Fixed-size tokens,
//! no escaped strings, canonical original IDs and bounded polls prevent JSON
//! byte length from hiding an unpriced decoded allocation. Output is the same
//! explicit-row schema as raw serde; raw serde helpers remain externally priced.
use super::*;
use std::io::{self, Write};

#[derive(Debug, Clone, Default, Serialize)]
pub struct NormalizedWireStats {
    pub input_byte_visits: usize,
    pub original_identity_visits: u64,
    pub parsed_row_visits: usize,
    pub validated_rows: usize,
    pub exported_rows: usize,
    pub encoded_bytes: usize,
    pub reserved_work: u64,
}
#[derive(Debug)]
pub struct NormalizedWireExport {
    bytes: Vec<u8>,
    stats: NormalizedWireStats,
    reservation: PersistentReservation,
}
impl NormalizedWireExport {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn stats(&self) -> &NormalizedWireStats {
        &self.stats
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
}

/// Safe encoded upper bound for fixed keys/enums, usize IDs, finite Ryu f64
/// numbers and 32 digest bytes. This is output sizing, never an expansion cap.
pub fn normalized_wire_size_bound(vocab: usize) -> Result<usize> {
    if vocab == 0 {
        return Err(DecoderRefusal::Shape("normalized wire vocabulary").into());
    }
    add(mul(vocab - 1, 128)?, 2048)
}

/// Work held before optional querying for output, bounded decode and complete
/// original replay. The byte bound is conservative output sizing; row expansion
/// still receives its own O(V) persistent reservation during decoding.
pub fn normalized_publication_work_bound(vocab: usize, hidden: usize) -> Result<u64> {
    let cap = normalized_wire_size_bound(vocab)?;
    let export = add(cap, add(mul(vocab, 2)?, 4096)?)?;
    let decode = add(
        mul(cap, 2)?,
        add(
            add(mul(vocab, hidden)?, vocab)?,
            add(mul(hidden, 16)?, add(mul(vocab, 4)?, 4096)?)?,
        )?,
    )?;
    (export as u64)
        .checked_add(decode as u64)
        .and_then(|work| work.checked_add(normalized_work_bound(vocab, hidden).ok()?))
        .ok_or(resource("normalized publication work overflow"))
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
    persistent: usize,
    scratch: usize,
    limits: &VocabOptions,
) -> Result<()> {
    let live = ledger.receipt().map_err(DecoderRefusal::Budget)?;
    let participating = add(
        live.live_persistent_bytes,
        owner.resident_bytes_outside_ledger(ledger),
    )?;
    if add(participating, persistent)? as u64 > limits.max_persistent_bytes
        || add(live.live_scratch_bytes, scratch)? as u64 > limits.max_scratch_bytes
    {
        return Err(resource("normalized aggregate wire capacity"));
    }
    Ok(())
}
fn write_error(lease: &WorkLease, limits: &VocabOptions) -> NormalizedHeadRefusal {
    checkpoint(lease, limits)
        .err()
        .map_or_else(evidence_error, Into::into)
}
fn evidence_error() -> NormalizedHeadRefusal {
    DecoderRefusal::Evidence("normalized wire format/header/coverage").into()
}

struct Parser<'a, 'b> {
    bytes: &'a [u8],
    pos: usize,
    lease: &'b WorkLease,
    limits: &'b VocabOptions,
}
impl<'a> Parser<'a, '_> {
    fn advance(&mut self) -> Result<()> {
        self.pos += 1;
        if self.pos.is_multiple_of(4096) {
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
            return Err(evidence_error());
        }
        self.advance()
    }
    fn take(&mut self, byte: u8) -> Result<bool> {
        self.ws()?;
        if self.bytes.get(self.pos) == Some(&byte) {
            self.advance()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn string(&mut self) -> Result<&'a str> {
        self.expect(b'"')?;
        let start = self.pos;
        loop {
            let byte = *self.bytes.get(self.pos).ok_or_else(evidence_error)?;
            if byte == b'"' {
                break;
            }
            // Fixed protocol tokens never require escaping or UTF-8 allocation.
            if !(32..=126).contains(&byte) || byte == b'\\' || self.pos - start >= 40 {
                return Err(evidence_error());
            }
            self.advance()?;
        }
        let end = self.pos;
        self.advance()?;
        std::str::from_utf8(&self.bytes[start..end]).map_err(|_| evidence_error())
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
                return Err(evidence_error());
            }
            self.advance()?;
        }
        if start == self.pos {
            return Err(evidence_error());
        }
        Ok(&self.bytes[start..self.pos])
    }
    fn uint(&mut self) -> Result<usize> {
        let token = self.number()?;
        if token.iter().any(|b| !b.is_ascii_digit()) || (token.len() > 1 && token[0] == b'0') {
            return Err(evidence_error());
        }
        std::str::from_utf8(token)
            .map_err(|_| evidence_error())?
            .parse()
            .map_err(|_| evidence_error())
    }
    fn float(&mut self) -> Result<f64> {
        let value: f64 = serde_json::from_slice(self.number()?).map_err(|_| evidence_error())?;
        if !value.is_finite() {
            return Err(evidence_error());
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
            *byte = u8::try_from(self.uint()?).map_err(|_| evidence_error())?;
        }
        self.expect(b']')?;
        Ok(digest)
    }
    fn row(&mut self) -> Result<NormalizedRowEvidence> {
        self.expect(b'{')?;
        let mut fields = 0u8;
        let (mut competitor, mut upper, mut method) =
            (0, 0.0, NormalizedSupportMethod::ProjectedDense);
        loop {
            let field = self.string()?;
            self.expect(b':')?;
            let bit = match field {
                "competitor" => {
                    competitor = self.uint()?;
                    1
                }
                "upper" => {
                    upper = self.float()?;
                    2
                }
                "method" => {
                    method = match self.string()? {
                        "ProjectedDense" => NormalizedSupportMethod::ProjectedDense,
                        "RadialTaylor" => NormalizedSupportMethod::RadialTaylor,
                        _ => return Err(evidence_error()),
                    };
                    4
                }
                _ => return Err(evidence_error()),
            };
            if fields & bit != 0 {
                return Err(evidence_error());
            }
            fields |= bit;
            if self.take(b'}')? {
                break;
            }
            self.expect(b',')?;
        }
        if fields != 7 {
            return Err(evidence_error());
        }
        Ok(NormalizedRowEvidence {
            competitor,
            upper,
            method,
        })
    }
    fn document(
        &mut self,
        vocab: usize,
        hidden: usize,
        target: usize,
        norm: FinalNormalizationKind,
        mut output: Option<&mut Vec<NormalizedRowEvidence>>,
    ) -> Result<[u8; 32]> {
        self.expect(b'{')?;
        let mut fields = 0u8;
        let mut digest = [0; 32];
        loop {
            let field = self.string()?;
            self.expect(b':')?;
            let bit = match field {
                "schema_version" => {
                    if self.uint()? != SCHEMA as usize {
                        return Err(evidence_error());
                    }
                    1
                }
                "scope_digest" => {
                    digest = self.digest()?;
                    2
                }
                "target" => {
                    if self.uint()? != target {
                        return Err(evidence_error());
                    }
                    4
                }
                "vocab_size" => {
                    if self.uint()? != vocab {
                        return Err(evidence_error());
                    }
                    8
                }
                "hidden_size" => {
                    if self.uint()? != hidden {
                        return Err(evidence_error());
                    }
                    16
                }
                "normalization" => {
                    let declared = match self.string()? {
                        "LayerNorm" => FinalNormalizationKind::LayerNorm,
                        "RmsNorm" => FinalNormalizationKind::RmsNorm,
                        _ => return Err(evidence_error()),
                    };
                    if declared != norm {
                        return Err(evidence_error());
                    }
                    32
                }
                "rows" => {
                    self.expect(b'[')?;
                    let mut count = 0usize;
                    let mut expected = 0usize;
                    if !self.take(b']')? {
                        loop {
                            if expected == target {
                                expected += 1;
                            }
                            // Refuse excess rows before decoding/pushing another allocation.
                            if expected >= vocab {
                                return Err(evidence_error());
                            }
                            let row = self.row()?;
                            if row.competitor != expected {
                                return Err(evidence_error());
                            }
                            if let Some(rows) = &mut output {
                                rows.push(row);
                            }
                            expected += 1;
                            count += 1;
                            if self.take(b']')? {
                                break;
                            }
                            self.expect(b',')?;
                        }
                    }
                    if count != vocab - 1 {
                        return Err(evidence_error());
                    }
                    64
                }
                _ => return Err(evidence_error()),
            };
            if fields & bit != 0 {
                return Err(evidence_error());
            }
            fields |= bit;
            if self.take(b'}')? {
                break;
            }
            self.expect(b',')?;
        }
        self.ws()?;
        if fields != 127 || self.pos != self.bytes.len() {
            return Err(evidence_error());
        }
        checkpoint(self.lease, self.limits)?;
        Ok(digest)
    }
}

#[derive(Debug)]
pub struct NormalizedWireReplay {
    pub report: NormalizedHeadReport,
    pub stats: NormalizedWireStats,
}

/// Decode and replay against the actual original owner, normalization and box.
/// O(V) explicit rows remain leased through the full original arithmetic replay.
pub fn replay_shared_normalized_json(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    bytes: &[u8],
    limits: &VocabOptions,
) -> Result<NormalizedWireReplay> {
    let ledger = owner.budget();
    ledger_checkpoint(&ledger, limits)?;
    ny_core::require_f64_interval_proof_environment()
        .map_err(|_| DecoderRefusal::Evidence("normalized binary64 proof environment"))?;
    let options = NormalizedHeadOptions {
        budget: &ledger,
        limits,
    };
    let head = owner.as_head();
    let _ = shape_with_owner(head, norm, pre, target, options, Some(owner))?;
    if bytes.len() > limits.max_evidence_bytes {
        return Err(resource("normalized wire input cap"));
    }
    let payload = add(
        mul(head.vocab_size - 1, size_of::<NormalizedRowEvidence>())?,
        4096,
    )?;
    let scratch = add(mul(head.hidden_size, 16)?, 16 * 1024)?;
    let work = add(
        mul(bytes.len(), 2)?,
        add(
            add(head.weights.len(), head.bias.len())?,
            add(
                mul(head.hidden_size, 16)?,
                add(mul(head.vocab_size, 4)?, 4096)?,
            )?,
        )?,
    )?;
    if payload as u64 > limits.max_persistent_bytes
        || scratch as u64 > limits.max_scratch_bytes
        || work as u64 > limits.max_terms
    {
        return Err(resource("normalized wire expansion prediction"));
    }
    live_capacity(&ledger, owner, payload, scratch, limits)?;
    let mut lease = ledger
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                persistent_bytes: payload,
                scratch_bytes: scratch,
                work: work as u64,
                proposals: 1,
            },
        )
        .map_err(DecoderRefusal::Budget)?;
    checkpoint(&lease, limits)?;
    let declared = Parser {
        bytes,
        pos: 0,
        lease: &lease,
        limits,
    }
    .document(head.vocab_size, head.hidden_size, target, norm.kind, None)?;
    let mut identity_stats = NormalizedHeadStats::default();
    let (checked_scope, actual_digest) = scope(
        head,
        norm,
        pre,
        target,
        &lease,
        limits,
        &mut identity_stats,
        Some(owner),
    )?;
    drop(checked_scope);
    if declared != actual_digest {
        return Err(DecoderRefusal::Evidence("normalized wire original scope binding").into());
    }
    let mut rows = reserve(head.vocab_size - 1)?;
    let repeated = Parser {
        bytes,
        pos: 0,
        lease: &lease,
        limits,
    }
    .document(
        head.vocab_size,
        head.hidden_size,
        target,
        norm.kind,
        Some(&mut rows),
    )?;
    if repeated != actual_digest {
        return Err(evidence_error());
    }
    if mul(rows.capacity(), size_of::<NormalizedRowEvidence>())? > payload {
        return Err(resource("normalized wire actual expansion capacity"));
    }
    let evidence = NormalizedHeadEvidence {
        schema_version: SCHEMA,
        scope_digest: actual_digest,
        target,
        vocab_size: head.vocab_size,
        hidden_size: head.hidden_size,
        normalization: norm.kind,
        rows,
    };
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let expansion = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
    let replayed =
        shared::replay_shared_normalized_next_token(owner, norm, pre, target, &evidence, limits)?;
    drop(evidence);
    drop(expansion);
    ledger_checkpoint(&ledger, limits)?;
    Ok(NormalizedWireReplay {
        report: replayed,
        stats: NormalizedWireStats {
            input_byte_visits: mul(bytes.len(), 2)?,
            original_identity_visits: identity_stats.original_head_visits
                + identity_stats.normalization_parameter_visits
                + identity_stats.region_visits,
            parsed_row_visits: mul(head.vocab_size - 1, 2)?,
            encoded_bytes: bytes.len(),
            reserved_work: work as u64,
            ..NormalizedWireStats::default()
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
            return Err(io::Error::other("normalized encoded cap"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Budget-aware plain-evidence export. The bytes must be independently replayed
/// before any publication can claim original-replay provenance.
pub fn export_shared_normalized_json(
    owner: &SharedAffineHead,
    evidence: &NormalizedHeadEvidence,
    limits: &VocabOptions,
) -> Result<NormalizedWireExport> {
    let head = owner.as_head();
    let ledger = owner.budget();
    if limits
        .budget
        .as_ref()
        .is_some_and(|b| !ledger.shares_ledger(b))
    {
        return Err(DecoderRefusal::Evidence("normalized export ledger substitution").into());
    }
    if evidence.schema_version != SCHEMA
        || evidence.vocab_size != head.vocab_size
        || evidence.hidden_size != head.hidden_size
        || evidence.target >= head.vocab_size
        || evidence.rows.len() != head.vocab_size - 1
    {
        return Err(evidence_error());
    }
    ledger_checkpoint(&ledger, limits)?;
    let cap = normalized_wire_size_bound(head.vocab_size)?;
    if cap > limits.max_evidence_bytes
        || cap as u64 > limits.max_persistent_bytes
        || limits.max_scratch_bytes < 16 * 1024
    {
        return Err(resource("normalized export prediction"));
    }
    let work = add(cap, add(mul(head.vocab_size, 2)?, 4096)?)?;
    if work as u64 > limits.max_terms {
        return Err(resource("normalized export work prediction"));
    }
    live_capacity(&ledger, owner, cap, 16 * 1024, limits)?;
    let mut lease = ledger
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                persistent_bytes: cap,
                scratch_bytes: 16 * 1024,
                work: work as u64,
                proposals: 1,
            },
        )
        .map_err(DecoderRefusal::Budget)?;
    let mut stats = NormalizedWireStats {
        reserved_work: work as u64,
        ..NormalizedWireStats::default()
    };
    let mut row_index: usize = 0;
    for id in 0..head.vocab_size {
        if id == evidence.target {
            continue;
        }
        if row_index.is_multiple_of(CHUNK) {
            checkpoint(&lease, limits)?;
        }
        let row = &evidence.rows[row_index];
        if row.competitor != id || !row.upper.is_finite() {
            return Err(evidence_error());
        }
        row_index += 1;
        stats.validated_rows += 1;
    }
    let mut writer = Writer {
        bytes: reserve(cap)?,
        cap,
        lease: &lease,
        limits,
    };
    write!(&mut writer, "{{\"schema_version\":1,\"scope_digest\":[")
        .map_err(|_| write_error(&lease, limits))?;
    for (i, value) in evidence.scope_digest.iter().enumerate() {
        if i != 0 {
            writer
                .write_all(b",")
                .map_err(|_| write_error(&lease, limits))?;
        }
        write!(&mut writer, "{value}").map_err(|_| write_error(&lease, limits))?;
    }
    write!(
        &mut writer,
        "],\"target\":{},\"vocab_size\":{},\"hidden_size\":{},\"normalization\":\"{}\",\"rows\":[",
        evidence.target,
        evidence.vocab_size,
        evidence.hidden_size,
        if evidence.normalization == FinalNormalizationKind::LayerNorm {
            "LayerNorm"
        } else {
            "RmsNorm"
        }
    )
    .map_err(|_| write_error(&lease, limits))?;
    for (i, row) in evidence.rows.iter().enumerate() {
        if i.is_multiple_of(CHUNK) {
            checkpoint(&lease, limits)?;
        }
        if i != 0 {
            writer
                .write_all(b",")
                .map_err(|_| write_error(&lease, limits))?;
        }
        serde_json::to_writer(&mut writer, row).map_err(|_| write_error(&lease, limits))?;
        stats.exported_rows += 1;
    }
    writer
        .write_all(b"]}")
        .map_err(|_| write_error(&lease, limits))?;
    stats.encoded_bytes = writer.bytes.len();
    let bytes = writer.bytes;
    if bytes.capacity() > cap {
        return Err(resource("normalized actual output capacity"));
    }
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let reservation = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
    ledger_checkpoint(&ledger, limits)?;
    Ok(NormalizedWireExport {
        bytes,
        stats,
        reservation,
    })
}

#[cfg(test)]
mod tests;
