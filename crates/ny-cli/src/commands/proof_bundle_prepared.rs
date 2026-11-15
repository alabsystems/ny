// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Export genuine prepared leaves into the existing independent v2 checker.
//! Only source names, original objective syntax and actual phase conjunctions
//! are exported. Bounds and source labels are hints: ny-proofcheck rederives
//! arithmetic and complete coverage from its own original-file readers.
//! All buffers overlap the authentic receipt on its original ledger. No
//! numerical arrays, compiled weights or ancestor gate frames are copied.

use super::*;
use std::{fmt::Write as _, mem::size_of, time::Instant};

use anyhow::{anyhow, bail, ensure, Result};
use ny_core::NyError;
use ny_propagate::certify::work::{LeaseClass, WorkRequest};

use crate::commands::prepared_root_publication::OriginalPreparedRootPublication;

#[derive(Clone, Copy, Debug)]
pub(super) struct ReplayQuote {
    pub(super) bytes: usize,
    work: u64,
    sites: usize,
    leaves: usize,
    literals: usize,
    model_bytes: usize,
    classes: usize,
}

/// What one exported line is, as the bundle writer's completion reads it
/// (the export's own `sites` / `clauses`, and what each cover node claims).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReplayKind {
    Sites,
    Clauses,
    /// A cover node: `whole_region` = no phase literal; `every_clause` = an
    /// emptiness close (`dis: "*"`), else it closes the original objective.
    Node {
        whole_region: bool,
        every_clause: bool,
    },
}

pub(super) struct ReplayRecord {
    pub(super) text: String,
    pub(super) kind: ReplayKind,
}

// Data drops before the same-ledger claim held by owner, even on channel,
// cancellation, formatting, I/O or detached-writer failures.
pub(super) struct ReplayPayload {
    pub(super) records: Vec<ReplayRecord>,
    pub(super) owner: Arc<RetainedBundleOwner>,
}

struct Site<'a> {
    op: usize,
    size: usize,
    name: &'a str,
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| anyhow!("prepared replay size overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("prepared replay size overflow"))
}
fn node_capacity(depth: usize, classes: usize) -> Result<usize> {
    add(256, add(mul(depth, 64)?, mul(classes, 32)?)?)
}
fn bitmap_words(outputs: usize) -> Result<usize> {
    Ok(add(outputs, 63)? / 64)
}
fn ny(error: anyhow::Error) -> NyError {
    NyError::InvalidSpec(format!("prepared replay: {error}"))
}
fn checkpoint(authority: &OriginalPreparedRootPublication, at: Instant) -> Result<()> {
    authority.budget().checkpoint_for(LeaseClass::Replay)?;
    ensure!(
        Instant::now() < at,
        "prepared replay export deadline expired"
    );
    Ok(())
}
fn poll(authority: &OriginalPreparedRootPublication, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(128) {
        checkpoint(authority, at)?;
    }
    Ok(())
}

pub(super) fn quote(authority: &OriginalPreparedRootPublication) -> Result<ReplayQuote> {
    authority.checkpoint(None)?;
    let stats = authority.replay_stats(None)?;
    let (model, _, _) = authority.replay_original_source(None)?;
    let classes = authority.classes().len();
    ensure!(
        stats.leaf_count > 0 && classes > 0,
        "prepared replay has no complete leaves/objective"
    );
    let site_text = add(256, add(mul(model.len(), 6)?, mul(stats.site_count, 128)?)?)?;
    let clauses_text = add(256, mul(classes, 128)?)?;
    let nodes_text = add(
        mul(stats.leaf_count, 256)?,
        add(
            mul(stats.literal_count, 64)?,
            mul(mul(stats.leaf_count, classes)?, 32)?,
        )?,
    )?;
    let records = mul(add(stats.leaf_count, 2)?, size_of::<ReplayRecord>())?;
    let sites = mul(stats.site_count, size_of::<Site<'static>>())?;
    let bitmap = mul(bitmap_words(add(classes, 1)?)?, size_of::<u64>())?;
    let bytes = add(
        32768,
        add(
            site_text,
            add(
                clauses_text,
                add(nodes_text, add(records, add(sites, bitmap)?)?)?,
            )?,
        )?,
    )?;
    // Independent admission for source/name scans, complete objective matching,
    // serialization and literal-to-original-site binary searches. Producer
    // census/traversal and full source checkpoints have their own work leases.
    let log_sites = usize::BITS as usize - stats.site_count.max(1).leading_zeros() as usize;
    let units = add(
        model.len(),
        add(
            mul(stats.leaf_count, classes)?,
            add(
                stats.site_count,
                mul(stats.literal_count, add(log_sites, 1)?)?,
            )?,
        )?,
    )?;
    let work = u64::try_from(add(8192, mul(units, 256)?)?)?;
    Ok(ReplayQuote {
        bytes,
        work,
        sites: stats.site_count,
        leaves: stats.leaf_count,
        literals: stats.literal_count,
        model_bytes: model.len(),
        classes,
    })
}

fn exact_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(n)?;
    ensure!(
        values.capacity() <= n,
        "prepared replay vector allocator exceeded admission"
    );
    Ok(values)
}
fn text(capacity: usize) -> Result<String> {
    let mut result = String::new();
    result.try_reserve_exact(capacity)?;
    ensure!(
        result.capacity() <= capacity,
        "prepared replay string allocator exceeded admission"
    );
    Ok(result)
}
fn string(
    out: &mut String,
    value: &str,
    authority: &OriginalPreparedRootPublication,
    at: Instant,
) -> Result<()> {
    out.push('"');
    for (i, ch) in value.chars().enumerate() {
        poll(authority, at, i)?;
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch < '\u{20}' => write!(out, "\\u{:04x}", ch as u32)?,
            ch => out.push(ch),
        }
    }
    out.push('"');
    Ok(())
}

/// Bounded borrowed protobuf walk used only for names. It computes no model
/// parameter, shape or bound. Numeric payloads are skipped without decoding or
/// allocation; the actual source and independent checker authenticate them.
struct Wire<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Wire<'a> {
    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for i in 0..10 {
            let byte = *self
                .bytes
                .get(self.at)
                .ok_or_else(|| anyhow!("truncated replay source varint"))?;
            self.at = add(self.at, 1)?;
            ensure!(i != 9 || byte & 0xfe == 0, "replay source varint overflow");
            value |= u64::from(byte & 127) << (7 * i);
            if byte & 128 == 0 {
                return Ok(value);
            }
        }
        bail!("overlong replay source varint")
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = add(self.at, n)?;
        let result = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| anyhow!("truncated replay source field"))?;
        self.at = end;
        Ok(result)
    }
    fn next(&mut self) -> Result<Option<(u64, u8, &'a [u8])>> {
        if self.at == self.bytes.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        ensure!(key >> 3 != 0, "zero replay source field");
        let wire = (key & 7) as u8;
        let payload = match wire {
            0 => {
                self.varint()?;
                &self.bytes[self.at..self.at]
            }
            1 => self.take(8)?,
            2 => {
                let n = usize::try_from(self.varint()?)?;
                self.take(n)?
            }
            5 => self.take(4)?,
            _ => bail!("unsupported replay source wire type"),
        };
        Ok(Some((key >> 3, wire, payload)))
    }
}

fn names<'a>(
    model: &'a [u8],
    sites: &mut [Site<'a>],
    authority: &OriginalPreparedRootPublication,
    at: Instant,
) -> Result<()> {
    let mut model_wire = Wire {
        bytes: model,
        at: 0,
    };
    let mut graph = None;
    let mut steps = 0usize;
    while let Some((field, wire, payload)) = model_wire.next()? {
        poll(authority, at, steps)?;
        steps = add(steps, 1)?;
        if field == 7 {
            ensure!(
                wire == 2 && graph.is_none(),
                "prepared replay requires one unambiguous original graph occurrence"
            );
            graph = Some(payload);
        }
    }
    let mut graph_wire = Wire {
        bytes: graph.ok_or_else(|| anyhow!("missing replay original graph"))?,
        at: 0,
    };
    let mut relus = 0usize;
    while let Some((field, wire, payload)) = graph_wire.next()? {
        poll(authority, at, steps)?;
        steps = add(steps, 1)?;
        if field != 1 {
            continue;
        }
        ensure!(wire == 2, "invalid replay original node wire type");
        let mut node = Wire {
            bytes: payload,
            at: 0,
        };
        let mut kind = None;
        let mut input = None;
        let mut input_count = 0usize;
        while let Some((field, wire, value)) = node.next()? {
            poll(authority, at, steps)?;
            steps = add(steps, 1)?;
            if matches!(field, 1 | 4) {
                ensure!(wire == 2, "invalid replay node name wire type");
                let value = std::str::from_utf8(value)?;
                if field == 1 {
                    input_count = add(input_count, 1)?;
                    input.get_or_insert(value);
                } else {
                    ensure!(kind.is_none(), "ambiguous replay original operator field");
                    kind = Some(value);
                }
            }
        }
        if kind == Some("Relu") {
            ensure!(input_count == 1, "ambiguous replay original ReLU input");
            let site = sites.get_mut(relus).ok_or_else(|| {
                anyhow!("extra original ReLU absent from authentic prepared topology")
            })?;
            site.name = input
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("empty replay original ReLU input"))?;
            relus = add(relus, 1)?;
        }
    }
    ensure!(
        relus == sites.len(),
        "original/prepared ReLU topology differs"
    );
    checkpoint(authority, at)
}

fn class_of(atom: &OutputConstraint, target: usize) -> Result<usize> {
    match *atom {
        OutputConstraint::LessEq(t, j) | OutputConstraint::LessThan(t, j) if t == target => Ok(j),
        OutputConstraint::GreaterEq(j, t) | OutputConstraint::GreaterThan(j, t) if t == target => {
            Ok(j)
        }
        _ => bail!("prepared replay original atom is outside authenticated target comparison"),
    }
}

pub(super) fn build(owner: Arc<RetainedBundleOwner>) -> Result<ReplayPayload> {
    let authority = owner
        .receipt
        .original_prepared_authority()
        .ok_or_else(|| anyhow!("prepared replay lacks genuine original owner"))?;
    authority.checkpoint(None)?;
    let quote = owner
        .replay_quote
        .ok_or_else(|| anyhow!("prepared replay lost its prepaid quote"))?;
    ensure!(
        owner._storage.shares_ledger(authority.budget()) && owner._storage.bytes() >= quote.bytes,
        "prepared replay lost original ledger/storage"
    );
    let (model, _, spec) = authority.replay_original_source(None)?;
    ensure!(
        model.len() == quote.model_bytes && authority.classes().len() == quote.classes,
        "prepared replay source quote changed"
    );
    let mut lease = authority.budget().lease_for(
        LeaseClass::Replay,
        WorkRequest {
            work: quote.work,
            ..WorkRequest::default()
        },
    )?;
    let at = lease
        .deadline()
        .map_or(authority.deadline(), |d| d.min(authority.deadline()));
    let mut records = exact_vec(add(quote.leaves, 2)?)?;
    let mut sites = exact_vec(quote.sites)?;
    authority.visit_replay_sites(Some(at), &mut |op, size| {
        if sites.len() >= quote.sites
            || sites.last().is_some_and(|s: &Site<'_>| s.op >= op)
            || size == 0
        {
            return Err(NyError::InvalidSpec(
                "invalid genuine replay site order/width".into(),
            ));
        }
        sites.push(Site { op, size, name: "" });
        Ok(())
    })?;
    ensure!(sites.len() == quote.sites, "partial prepared replay sites");
    names(model, &mut sites, &authority, at)?;
    let mut site_line = text(add(
        256,
        add(mul(model.len(), 6)?, mul(sites.len(), 128)?)?,
    )?)?;
    site_line.push_str("{\"kind\":\"sites\",\"tensors\":[");
    for (i, site) in sites.iter().enumerate() {
        poll(&authority, at, i)?;
        if i != 0 {
            site_line.push(',');
        }
        write!(site_line, "{{\"id\":{},\"name\":", site.op)?;
        string(&mut site_line, site.name, &authority, at)?;
        write!(site_line, ",\"size\":{},\"op\":{}}}", site.size, site.op)?;
    }
    site_line.push_str("]}");
    records.push(ReplayRecord {
        text: site_line,
        kind: ReplayKind::Sites,
    });
    let outputs = add(quote.classes, 1)?;
    ensure!(
        authority.target() < outputs,
        "prepared replay target exceeds full objective"
    );
    let mut seen = exact_vec(bitmap_words(outputs)?)?;
    seen.resize(bitmap_words(outputs)?, 0u64);
    let mut clause_line = text(add(256, mul(quote.classes, 128)?)?)?;
    clause_line.push_str("{\"kind\":\"clauses\",\"region\":0,\"list\":[");
    let clause_count = if spec.is_disjunction {
        spec.output_constraint_clauses.len()
    } else {
        1
    };
    ensure!(
        clause_count == quote.classes,
        "prepared replay parsed clause count changed"
    );
    for i in 0..clause_count {
        poll(&authority, at, i)?;
        let atoms = if spec.is_disjunction {
            spec.output_constraint_clauses[i].as_slice()
        } else {
            spec.output_constraints.as_slice()
        };
        ensure!(
            atoms.len() == 1,
            "prepared replay requires exact original singleton clauses"
        );
        let class = class_of(&atoms[0], authority.target())?;
        ensure!(
            class < outputs && class != authority.target(),
            "foreign prepared replay original class"
        );
        let bit = 1u64 << (class % 64);
        ensure!(
            seen[class / 64] & bit == 0,
            "duplicate prepared replay original class"
        );
        seen[class / 64] |= bit;
        let canonical = match atoms[0] {
            OutputConstraint::LessThan(_, _) | OutputConstraint::GreaterThan(_, _) => "<0",
            _ => "<=0",
        };
        if i != 0 {
            clause_line.push(',');
        }
        write!(clause_line, "{{\"id\":{class},\"atoms\":[\"")?;
        if authority.target() < class {
            write!(
                clause_line,
                "Y{}:1,Y{class}:-1{canonical}",
                authority.target()
            )?;
        } else {
            write!(
                clause_line,
                "Y{class}:-1,Y{}:1{canonical}",
                authority.target()
            )?;
        }
        clause_line.push_str("\"]}");
    }
    for (i, &class) in authority.classes().iter().enumerate() {
        poll(&authority, at, i)?;
        ensure!(
            class < outputs && seen[class / 64] & (1u64 << (class % 64)) != 0,
            "parsed original replay omits a required class"
        );
    }
    clause_line.push_str("]}");
    records.push(ReplayRecord {
        text: clause_line,
        kind: ReplayKind::Clauses,
    });
    let mut leaves = 0usize;
    let mut literals = 0usize;
    authority.visit_replay_leaves(Some(at), &mut |leaf| {
        let result = (|| -> Result<()> {
            checkpoint(&authority, at)?;
            ensure!(
                leaves < quote.leaves && leaf.classes() == authority.classes(),
                "partial/foreign prepared replay leaf objective"
            );
            let depth = leaf.literal_count();
            let mut line = text(node_capacity(
                depth,
                if leaf.is_empty() { 0 } else { quote.classes },
            )?)?;
            write!(
                line,
                "{{\"kind\":\"node\",\"id\":{leaves},\"region\":0,\"lits\":["
            )?;
            let mut count = 0usize;
            for item in leaf.literals() {
                let literal = item?;
                poll(&authority, at, count)?;
                let pos = sites
                    .binary_search_by_key(&literal.op(), |s| s.op)
                    .map_err(|_| anyhow!("leaf literal is not an original authenticated ReLU"))?;
                ensure!(
                    literal.index() < sites[pos].size && count < depth,
                    "prepared replay literal exceeds original row"
                );
                if count != 0 {
                    line.push(',');
                }
                write!(
                    line,
                    "[{},{},{}]",
                    literal.op(),
                    literal.index(),
                    u8::from(literal.positive())
                )?;
                count = add(count, 1)?;
            }
            ensure!(count == depth, "partial prepared replay phase ancestry");
            literals = add(literals, count)?;
            line.push_str("],\"close\":");
            line.push_str(if leaf.is_empty() {
                "\"empty\""
            } else if count == 0 {
                "\"root\""
            } else {
                "\"bound\""
            });
            line.push_str(",\"src\":\"original_prepared_root\",\"dis\":");
            if leaf.is_empty() {
                line.push_str("\"*\"");
            } else {
                ensure!(
                    leaf.min_margin().is_finite() && leaf.min_margin() > 0.0,
                    "prepared replay actual bound leaf is not closed"
                );
                line.push('[');
                for (i, &class) in authority.classes().iter().enumerate() {
                    poll(&authority, at, i)?;
                    if i != 0 {
                        line.push(',');
                    }
                    write!(line, "{class}")?;
                }
                line.push(']');
            }
            // No cached scalar is used by the offline arithmetic checker.
            line.push_str(",\"hint\":null}");
            ensure!(
                line.len() <= line.capacity(),
                "prepared replay line exceeds admission"
            );
            records.push(ReplayRecord {
                text: line,
                kind: ReplayKind::Node {
                    whole_region: count == 0,
                    every_clause: leaf.is_empty(),
                },
            });
            leaves = add(leaves, 1)?;
            checkpoint(&authority, at)
        })();
        result.map_err(ny)
    })?;
    ensure!(
        leaves == quote.leaves && literals == quote.literals,
        "prepared replay incomplete original cover"
    );
    let mut actual = mul(records.capacity(), size_of::<ReplayRecord>())?;
    for (i, record) in records.iter().enumerate() {
        poll(&authority, at, i)?;
        actual = add(actual, record.text.capacity())?;
    }
    ensure!(
        actual <= quote.bytes,
        "prepared replay payload allocator exceeded admission"
    );
    drop((seen, sites));
    lease.complete()?;
    authority.checkpoint(Some(at))?;
    Ok(ReplayPayload { records, owner })
}
