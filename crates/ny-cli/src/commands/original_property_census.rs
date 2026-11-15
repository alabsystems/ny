// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Paid parsing of the original, flat complete robustness property.
//!
//! The one MiB limit bounds grammar alone. Admission still spends the caller's
//! unchanged ledger. A constant-memory scan counts actual tokens, list slots,
//! declared axes, decimal digits/scales, and output atoms BEFORE either parser
//! allocates. The source-authentication pass then excludes nested Boolean
//! products, arithmetic and aliases before the ordinary parser is invoked.
//!
//! Transient figures are source-derived capacity BOUNDS, not measured peaks:
//! syntax.rs and normalize/convert.rs can hold two token/Expr copies; the
//! certified_input_box.rs pass holds one borrowed RawExpr tree, two rational
//! endpoint arrays and one coordinate's conversion temporaries. Normalization
//! of this authenticated surface has singleton DNF clauses only. Retained
//! public vector capacities are checked explicitly. Ceased parser storage is
//! dropped before its scratch claim releases. Vec growth bounds are pinned to
//! this repository's std/parser implementation, not a portable allocator API.

use std::{collections::BTreeMap, mem::size_of, time::Instant};

use anyhow::{anyhow, ensure, Result};
use ny_onnx::vnnlib::{parse_vnnlib_with_certified_input_box, OutputConstraint};
use ny_propagate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkRequest};
use sha2::{Digest, Sha256};

use super::prepared_root_publication::{
    authenticate_raw_objective_with_cap, cross_check, validate_literal, Atom,
    OriginalInputEnclosure, OriginalProperty,
};

pub(super) const MAX_STAGED_PROPERTY_BYTES: usize = 1 << 20;
const META: usize = 8192;
const POLL: usize = 4096;
const MAX_DEPTH: usize = 5;
// Expr and RawExpr have String/f64/Vec or &str/Vec payloads. 64 covers their
// enum/header layout; every dynamic child slot is counted separately below.
const EXPR_SLOT_BYTES: usize = 64;
// BigInt is a sign + Vec of machine digits; two BigInts form BigRational.
// This bound includes Option/enum/header padding, excluding counted limbs.
const RATIONAL_HEADER_BYTES: usize = 96;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct OriginalPropertyAllocationReceipt {
    pub(super) source_bytes: usize,
    pub(super) token_count: usize,
    pub(super) list_count: usize,
    pub(super) expression_capacity_slots_bound: usize,
    pub(super) input_count: usize,
    pub(super) output_count: usize,
    pub(super) decimal_count: usize,
    pub(super) max_decimal_digits: usize,
    pub(super) max_decimal_scale: usize,
    pub(super) parser_scratch_capacity_bound: usize,
    pub(super) retained_capacity_bound: usize,
    pub(super) retained_visible_bytes: usize,
}

pub(super) struct StagedOriginalProperty {
    property: OriginalProperty,
    source_resident: PersistentReservation,
    resident: PersistentReservation,
    receipt: OriginalPropertyAllocationReceipt,
    deadline: Instant,
}
impl StagedOriginalProperty {
    pub(super) fn into_parts(
        self,
    ) -> (
        OriginalProperty,
        PersistentReservation,
        PersistentReservation,
        OriginalPropertyAllocationReceipt,
        Instant,
    ) {
        (
            self.property,
            self.source_resident,
            self.resident,
            self.receipt,
            self.deadline,
        )
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Census {
    bytes: usize,
    tokens: usize,
    token_bytes: usize,
    lists: usize,
    list_slots: usize,
    inputs: usize,
    outputs: usize,
    decimals: usize,
    max_digits: usize,
    max_scale: usize,
    literal_work: usize,
    comparison_work: usize,
    conversion_work: usize,
}
impl Census {
    fn retained(self) -> Result<usize> {
        let rows = self.outputs.saturating_sub(1);
        let row_capacity = growth(rows)?;
        // Spec input_bounds and declared_input_bounds plus the actual moved
        // certified lower/upper arrays: no second endpoint copy is made.
        let inputs = mul(self.inputs, 48)?;
        // Flat constraints, outer singleton clauses, singleton inner Vec
        // capacities, empty per-clause BTreeMap headers, raw atoms and classes.
        let outputs = add(
            mul(
                row_capacity,
                add(
                    mul(5, size_of::<OutputConstraint>())?,
                    add(
                        size_of::<Vec<OutputConstraint>>(),
                        add(size_of::<BTreeMap<usize, (f64, f64)>>(), size_of::<Atom>())?,
                    )?,
                )?,
            )?,
            mul(rows, size_of::<usize>())?,
        )?;
        add(add(inputs, outputs)?, META)
    }

    fn scratch(self) -> Result<usize> {
        // Ordinary parser's own tree remains live through a separately
        // tokenized normalization pass. Both cleaned strings, token Vecs,
        // token and Expr symbol copies, current token scratch and all child
        // capacities overlap. The later exact parse needs no larger tree.
        let one_syntax = add(
            mul(add(self.bytes, 1)?, 2)?,
            add(
                mul(growth(self.tokens)?, size_of::<String>())?,
                add(
                    mul(add(self.token_bytes, mul(self.tokens, 4)?)?, 4)?,
                    mul(self.list_slots, EXPR_SLOT_BYTES)?,
                )?,
            )?,
        )?;
        // hashbrown lower/upper maps: at most one entry per declared axis,
        // each map includes bucket slack, control bytes and usize/f64 pairs.
        // Authenticated scalar declarations create no tensor declaration map.
        let input_maps = mul(self.inputs, 256)?;
        // max decimal numerator/denominator bits: 10^k < 2^(4k).
        // pow/parse limb Vec slack <=4; each bound retains two BigInts. Exact
        // tightening stores just two rationals per axis, even with repeated
        // input assertions. Per-coordinate conversion/center/radius temporaries
        // are separate, bounded by four decimal widths plus two binary64
        // denominators (1074 bits each), with 32 simultaneous limb buffers.
        let bits = mul(add(self.max_digits, self.max_scale)?, 4)?;
        let limb = mul(add(bits.div_ceil(8), 16)?, 4)?;
        let endpoints = mul(
            mul(self.inputs, 2)?,
            add(RATIONAL_HEADER_BYTES, mul(limb, 2)?)?,
        )?;
        let conversion_bits = add(mul(bits, 4)?, 2200)?;
        let conversion = mul(mul(add(conversion_bits.div_ceil(8), 32)?, 4)?, 32)?;
        // Certified six f64 planes + point mask; its two endpoints move to
        // persistent ownership. Four center/radius lanes grow by push.
        let certified = add(mul(self.inputs, 17)?, mul(growth(self.inputs)?, 32)?)?;
        // Direct normalization has <=2 variables/map and <=one atom/clause.
        // 8 simultaneous map/boolean/clause copies per output atom, each
        // <=2048 bytes including BTree leaf, 4-slot Vec and conversion scratch.
        // Input-only linear maps are transient for one atom, paid by META.
        let normalizer = mul(self.outputs, 16384)?;
        add(
            mul(one_syntax, 2)?,
            add(
                input_maps,
                add(
                    endpoints,
                    add(conversion, add(certified, add(normalizer, META)?)?)?,
                )?,
            )?,
        )
    }

    fn work(self) -> Result<u64> {
        // Linear scans/token copies/tree traversals; exact decimal gcd/division
        // and center conversion are charged at a conservative quadratic limb
        // envelope. Binary64 conversion adds <=2200bits per coordinate.
        // Class duplicate detection is genuinely quadratic and charged here.
        let rational_work = add(
            self.literal_work,
            add(self.comparison_work, self.conversion_work)?,
        )?;
        let work = add(
            mul(self.bytes, 128)?,
            add(
                rational_work,
                add(mul(mul(self.outputs, self.outputs)?, 32)?, META)?,
            )?,
        )?;
        u64::try_from(work).map_err(Into::into)
    }
}

/// `source_claim` is the independently retained snapshot claim, already paid
/// before file reading. The staged property claim pays spec/objective/domain.
pub(super) fn parse_staged_original_property(
    source: String,
    source_claim: &PersistentReservation,
    budget: &WorkBudget,
    requested: Instant,
) -> Result<StagedOriginalProperty> {
    checkpoint(budget, requested)?;
    ensure!(
        source_claim.shares_ledger(budget) && source_claim.bytes() >= source.capacity(),
        "staged property requires its original paid source snapshot"
    );
    let census = census(&source, budget, requested)?;
    let scratch = census.scratch()?;
    let persistent = census.retained()?;
    let mut lease = budget.lease_for(
        LeaseClass::Fallback,
        WorkRequest {
            scratch_bytes: scratch,
            persistent_bytes: persistent,
            work: census.work()?,
            ..WorkRequest::default()
        },
    )?;
    let at = lease.deadline().map_or(requested, |d| d.min(requested));
    checkpoint(budget, at)?;
    // No unsupported tree reaches the private ordinary/exact parser merely
    // because its token counts fit. Raw semantics are authenticated first.
    let objective =
        authenticate_raw_objective_with_cap(&source, budget, at, MAX_STAGED_PROPERTY_BYTES)?;
    ensure!(
        objective.inputs == census.inputs && objective.outputs == census.outputs,
        "staged property declaration census disagrees with raw source"
    );
    let (spec, input) = parse_vnnlib_with_certified_input_box(&source)?;
    checkpoint(budget, at)?;
    cross_check(&objective, &spec, &input)?;
    // Consuming this genuine source certificate moves the actual endpoint
    // arrays; its unused point/center/radius arrays cease here.
    let (lower, upper, points) = input.into_parts();
    drop(points);
    let input = OriginalInputEnclosure { lower, upper };
    let mut sha = Sha256::new();
    for block in source.as_bytes().chunks(POLL) {
        checkpoint(budget, at)?;
        sha.update(block);
    }
    let property = OriginalProperty {
        source,
        spec,
        input,
        objective,
        sha256: sha.finalize().into(),
    };
    let visible = retained_visible(&property, budget, at)?;
    ensure!(
        visible <= persistent,
        "actual retained property capacity exceeds grammar admission"
    );
    lease.mark_materialized()?;
    let resident = lease.commit_persistent()?;
    let receipt = OriginalPropertyAllocationReceipt {
        source_bytes: census.bytes,
        token_count: census.tokens,
        list_count: census.lists,
        expression_capacity_slots_bound: census.list_slots,
        input_count: census.inputs,
        output_count: census.outputs,
        decimal_count: census.decimals,
        max_decimal_digits: census.max_digits,
        max_decimal_scale: census.max_scale,
        parser_scratch_capacity_bound: scratch,
        retained_capacity_bound: persistent,
        retained_visible_bytes: visible,
    };
    checkpoint(budget, at)?;
    Ok(StagedOriginalProperty {
        property,
        source_resident: source_claim.clone(),
        resident,
        receipt,
        deadline: at,
    })
}

fn census(text: &str, budget: &WorkBudget, requested: Instant) -> Result<Census> {
    checkpoint(budget, requested)?;
    ensure!(
        text.len() <= MAX_STAGED_PROPERTY_BYTES,
        "staged original property grammar exceeds one MiB"
    );
    let mut lease = budget.lease_for(
        LeaseClass::Fallback,
        WorkRequest {
            scratch_bytes: META,
            work: u64::try_from(add(mul(text.len(), 32)?, META)?)?,
            ..WorkRequest::default()
        },
    )?;
    let at = lease.deadline().map_or(requested, |d| d.min(requested));
    let mut out = Census {
        bytes: text.len(),
        ..Census::default()
    };
    let mut cursor = Scan {
        text,
        pos: 0,
        budget,
        at,
    };
    let mut items = [0usize; MAX_DEPTH];
    let mut depth = 0usize;
    let mut roots = 0usize;
    let mut declaration = false;
    while let Some(token) = cursor.token()? {
        out.tokens = add(out.tokens, 1)?;
        out.token_bytes = add(out.token_bytes, token.len())?;
        match token {
            "(" => {
                ensure!(depth < MAX_DEPTH, "staged flat property nesting exceeded");
                items[depth] = 0;
                depth += 1;
                out.lists = add(out.lists, 1)?;
            }
            ")" => {
                ensure!(depth > 0, "staged property unmatched close");
                depth -= 1;
                out.list_slots = add(out.list_slots, growth(items[depth])?)?;
                if depth == 0 {
                    roots = add(roots, 1)?;
                } else {
                    items[depth - 1] = add(items[depth - 1], 1)?;
                }
            }
            atom => {
                ensure!(
                    depth > 0 && atom.is_ascii() && atom.len() <= 64,
                    "staged property atom is outside bounded flat grammar"
                );
                items[depth - 1] = add(items[depth - 1], 1)?;
                if declaration {
                    if let Some(index) = canonical(atom, "X_") {
                        ensure!(
                            index == out.inputs,
                            "staged property input declarations are not ordered"
                        );
                        out.inputs = add(out.inputs, 1)?;
                    } else if let Some(index) = canonical(atom, "Y_") {
                        ensure!(
                            index == out.outputs,
                            "staged property output declarations are not ordered"
                        );
                        out.outputs = add(out.outputs, 1)?;
                    } else {
                        ensure!(false, "staged property declaration is not canonical");
                    }
                    declaration = false;
                } else if matches!(atom, "declare-const" | "declare-fun") {
                    declaration = true;
                }
                if atom
                    .as_bytes()
                    .first()
                    .is_some_and(|&c| c.is_ascii_digit() || matches!(c, b'+' | b'-' | b'.'))
                {
                    validate_literal(atom)?;
                    let width = decimal_width(atom)?;
                    out.decimals = add(out.decimals, 1)?;
                    out.max_digits = out.max_digits.max(width.digits);
                    out.max_scale = out.max_scale.max(width.scale);
                    out.literal_work = add(out.literal_work, rational_work(width, 64)?)?;
                }
            }
        }
    }
    ensure!(
        depth == 0 && !declaration && out.inputs > 0 && out.outputs >= 2,
        "staged property lacks complete bounded declarations"
    );
    out.list_slots = add(out.list_slots, growth(roots)?)?;
    lease.complete()?;
    drop(lease);
    coordinate_work(text, &mut out, budget, at)?;
    Ok(out)
}

#[derive(Clone, Copy, Default)]
struct DecimalWidth {
    digits: usize,
    scale: usize,
}

fn decimal_width(atom: &str) -> Result<DecimalWidth> {
    let unsigned = atom.trim_start_matches(['+', '-']);
    let (mantissa, exponent) = unsigned
        .split_once(['e', 'E'])
        .map_or(Ok((unsigned, 0i32)), |(m, e)| {
            e.parse::<i32>().map(|e| (m, e))
        })?;
    let fraction = mantissa.split_once('.').map_or(0, |(_, f)| f.len());
    Ok(DecimalWidth {
        digits: mantissa.bytes().filter(|b| b.is_ascii_digit()).count(),
        scale: i32::try_from(fraction)?
            .checked_sub(exponent)
            .ok_or_else(|| anyhow!("staged literal scale overflow"))?
            .unsigned_abs() as usize,
    })
}

fn rational_work(width: DecimalWidth, extra_bits: usize) -> Result<usize> {
    // 10^k < 2^(4k). Sixteen bits per decimal digit/scale retain fourfold
    // bit slack for the pinned parse/pow/GCD/division/continued-fraction
    // comparison envelope. Parsing/comparing decimal rationals does not
    // manufacture a binary64 denominator. The original 2200-bit allowance
    // remains on each completed coordinate conversion, including DD residuals.
    let limbs = add(mul(add(width.digits, width.scale)?, 16)?, extra_bits)?.div_ceil(32);
    mul(mul(limbs, limbs)?, 256)
}

fn coordinate_work(
    text: &str,
    out: &mut Census,
    budget: &WorkBudget,
    requested: Instant,
) -> Result<()> {
    // This is cost metadata, never an input certificate. Price its complete
    // capacity and second grammar scan before allocating or inspecting IDs.
    let mut lease = budget.lease_for(
        LeaseClass::Fallback,
        WorkRequest {
            scratch_bytes: add(mul(out.inputs, size_of::<DecimalWidth>())?, META)?,
            work: u64::try_from(add(mul(text.len(), 32)?, mul(out.inputs, 128)?)?)?,
            ..WorkRequest::default()
        },
    )?;
    let at = lease.deadline().map_or(requested, |d| d.min(requested));
    let mut widths = Vec::new();
    widths.try_reserve_exact(out.inputs)?;
    ensure!(
        widths.capacity() <= out.inputs,
        "staged coordinate cost allocator exceeds admission"
    );
    for i in 0..out.inputs {
        poll(budget, at, i)?;
        widths.push(DecimalWidth::default());
    }
    let mut cursor = Scan {
        text,
        pos: 0,
        budget,
        at,
    };
    let mut depth = 0usize;
    let mut input: Option<usize> = None;
    let mut width: Option<DecimalWidth> = None;
    let mut relation: Option<&str> = None;
    while let Some(token) = cursor.token()? {
        match token {
            "(" => {
                depth = add(depth, 1)?;
                if depth == 2 {
                    input = None;
                    width = None;
                    relation = None;
                }
            }
            ")" => {
                if depth == 2 {
                    if let (Some(index), Some(value)) = (input, width) {
                        let previous = widths.get_mut(index).ok_or_else(|| {
                            anyhow!("staged cost references undeclared coordinate")
                        })?;
                        previous.digits = previous.digits.max(value.digits);
                        previous.scale = previous.scale.max(value.scale);
                        // Tightening compares with the previous retained rational;
                        // a later short spelling cannot erase an earlier large one.
                        // Equality can tighten both endpoint slots.
                        let comparisons = if relation == Some("=") { 2 } else { 1 };
                        out.comparison_work = add(
                            out.comparison_work,
                            mul(rational_work(*previous, 64)?, comparisons)?,
                        )?;
                    }
                }
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| anyhow!("staged cost unmatched close"))?;
            }
            atom if depth == 2 => {
                if relation.is_none() {
                    relation = Some(atom);
                }
                if let Some(index) = canonical(atom, "X_") {
                    ensure!(
                        index < out.inputs,
                        "staged cost references undeclared coordinate"
                    );
                    input = Some(index);
                } else if atom
                    .as_bytes()
                    .first()
                    .is_some_and(|&c| c.is_ascii_digit() || matches!(c, b'+' | b'-' | b'.'))
                {
                    width = Some(decimal_width(atom)?);
                }
            }
            _ => (),
        }
    }
    for (i, width) in widths.iter().enumerate() {
        poll(budget, at, i)?;
        out.conversion_work = add(out.conversion_work, rational_work(*width, 2200)?)?;
    }
    checkpoint(budget, at)?;
    drop(widths);
    lease.complete()?;
    Ok(())
}

struct Scan<'a> {
    text: &'a str,
    pos: usize,
    budget: &'a WorkBudget,
    at: Instant,
}
impl<'a> Scan<'a> {
    fn token(&mut self) -> Result<Option<&'a str>> {
        let bytes = self.text.as_bytes();
        while self.pos < bytes.len() {
            poll(self.budget, self.at, self.pos)?;
            if matches!(bytes[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else if bytes[self.pos] == b';' {
                while self.pos < bytes.len() && bytes[self.pos] != b'\n' {
                    poll(self.budget, self.at, self.pos)?;
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
        if self.pos == bytes.len() {
            return Ok(None);
        }
        let start = self.pos;
        self.pos += 1;
        if !matches!(bytes[start], b'(' | b')') {
            while self.pos < bytes.len()
                && !matches!(
                    bytes[self.pos],
                    b' ' | b'\t' | b'\r' | b'\n' | b'(' | b')' | b';'
                )
            {
                poll(self.budget, self.at, self.pos)?;
                self.pos += 1;
            }
        }
        // ASCII tokens only: Unicode comments are skipped without slicing
        // inside a character, and non-ASCII semantic atoms fail closed.
        ensure!(
            bytes[start..self.pos].is_ascii(),
            "staged non-ASCII semantic token"
        );
        Ok(Some(&self.text[start..self.pos]))
    }
}

fn canonical(token: &str, prefix: &str) -> Option<usize> {
    let digits = token.strip_prefix(prefix)?;
    if digits.is_empty()
        || !digits.bytes().all(|b| b.is_ascii_digit())
        || (digits.len() > 1 && digits.starts_with('0'))
    {
        return None;
    }
    digits.parse().ok()
}
fn growth(n: usize) -> Result<usize> {
    if n == 0 {
        return Ok(0);
    }
    n.max(4)
        .checked_next_power_of_two()
        .ok_or_else(|| anyhow!("property capacity overflow"))
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| anyhow!("property byte overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("property byte overflow"))
}
fn checkpoint(budget: &WorkBudget, at: Instant) -> Result<()> {
    budget.checkpoint_for(LeaseClass::Fallback)?;
    ensure!(
        Instant::now() < at,
        "staged property original deadline expired"
    );
    ny_core::require_f64_interval_proof_environment()?;
    Ok(())
}
fn poll(budget: &WorkBudget, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(budget, at)?;
    }
    Ok(())
}
fn retained_visible(
    property: &OriginalProperty,
    budget: &WorkBudget,
    at: Instant,
) -> Result<usize> {
    let spec = &property.spec;
    let mut bytes = META;
    for n in [
        mul(spec.input_bounds.capacity(), size_of::<(f64, f64)>())?,
        mul(
            spec.declared_input_bounds.capacity(),
            size_of::<(f64, f64)>(),
        )?,
        mul(
            spec.output_constraints.capacity(),
            size_of::<OutputConstraint>(),
        )?,
        mul(
            spec.output_constraint_clauses.capacity(),
            size_of::<Vec<OutputConstraint>>(),
        )?,
        mul(
            spec.per_clause_input_bounds.capacity(),
            size_of::<BTreeMap<usize, (f64, f64)>>(),
        )?,
        mul(property.objective.classes.capacity(), size_of::<usize>())?,
        mul(property.objective.atoms.capacity(), size_of::<Atom>())?,
        mul(property.input.lower.capacity(), 8)?,
        mul(property.input.upper.capacity(), 8)?,
    ] {
        bytes = add(bytes, n)?;
    }
    for (i, clause) in spec.output_constraint_clauses.iter().enumerate() {
        poll(budget, at, i)?;
        bytes = add(
            bytes,
            mul(clause.capacity(), size_of::<OutputConstraint>())?,
        )?;
    }
    ensure!(
        spec.version.is_none()
            && spec.dual_network.is_none()
            && spec.per_clause_input_bounds.iter().all(BTreeMap::is_empty),
        "staged property retained unsupported metadata"
    );
    checkpoint(budget, at)?;
    Ok(bytes)
}

#[cfg(test)]
#[path = "original_property_census_tests.rs"]
mod tests;
