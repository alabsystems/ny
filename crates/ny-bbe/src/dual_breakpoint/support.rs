// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Request-local support compilation for UNTRUSTED proposals. The supplied
//! reduced costs already contain the depth's enc substitution and fixed phase
//! offsets. Every active row participates, even at zero optimizer multipliers.
//! No summary survives the request and no coefficient matrix is copied.
//!
//! Equal/opposite signatures are checked against ALL original stored floats;
//! hashes only choose comparisons. Sign folding uses directed bounds over the
//! entire permitted multiplier box, including inherited values above the search
//! ceiling. Changed summation order is search arithmetic, not a certificate.

use std::collections::HashMap;

use ny_core::search::{DualSearchRequest, SupportSearchStats};

use super::{poll, MULTIPLIER_LIMIT};

const MIN_COLUMNS: usize = 64;
const MAX_ROW_TERMS: usize = 250_000;
const MAX_BUILD_READS: u64 = 1_000_000;
const MAX_HASHES: usize = 1024;

#[derive(Clone, Copy)]
struct Group {
    original: usize,
    weight: usize,
}

pub(super) struct Support {
    groups: Option<Vec<Group>>,
    folded_dot: Vec<f64>,
    folded_norm: f64,
}

impl Support {
    pub(super) fn dense() -> Self {
        Self {
            groups: None,
            folded_dot: Vec::new(),
            folded_norm: 0.0,
        }
    }

    pub(super) fn len(&self, original: usize) -> usize {
        self.groups.as_ref().map_or(original, Vec::len)
    }

    pub(super) fn original(&self, j: usize) -> usize {
        self.groups.as_ref().map_or(j, |g| g[j].original)
    }

    pub(super) fn weight(&self, j: usize) -> usize {
        self.groups.as_ref().map_or(1, |g| g[j].weight)
    }

    pub(super) fn dot(&self, row: usize) -> f64 {
        self.folded_dot.get(row).copied().unwrap_or(0.0)
    }

    pub(super) fn norm(&self) -> f64 {
        self.folded_norm
    }

    pub(super) fn bytes(&self) -> u64 {
        self.groups.as_ref().map_or(0, |g| g.capacity() as u64 * 16)
            + self.folded_dot.capacity() as u64 * 8
    }

    fn search_bytes(&self, n: usize, depth: usize) -> u64 {
        self.bytes()
            .saturating_add((self.len(n) as u64).saturating_mul(40))
            .saturating_add((depth as u64).saturating_mul(48))
            .saturating_add(1024)
    }

    /// Bounded automatic pilot. A decline keeps the dense representation. It
    /// may have spent bounded work; its receipt preserves that cost too.
    pub(super) fn compile(req: &DualSearchRequest<'_>, receipt: &mut SupportSearchStats) -> Self {
        let n = req.rc.len();
        let depth = req.rows.len();
        receipt.input_columns = n as u64;
        receipt.active_columns = n as u64;
        receipt.dense_bytes = (n as u64)
            .saturating_mul(40)
            .saturating_add((depth as u64).saturating_mul(48))
            .saturating_add(1024);
        if req.steps == 0
            || n < MIN_COLUMNS
            || n.saturating_mul(depth) > MAX_ROW_TERMS
            || compiler_admission_bytes(n, depth) > req.mem_cap_bytes
            || poll(req.deadline).is_err()
        {
            receipt.fallback = 1;
            return Self::dense();
        }
        let Some(view) = Self::try_compile(req, receipt) else {
            receipt.fallback = 1;
            receipt.untouched_columns = 0;
            receipt.grouped_columns = 0;
            receipt.sign_folded_columns = 0;
            return Self::dense();
        };
        let active = view.len(n);
        // The pilot must remove substantial row work, rather than retain a
        // second full-column vector for negligible compression.
        if active.saturating_mul(5) > n.saturating_mul(3)
            || view.search_bytes(n, depth) > req.mem_cap_bytes
        {
            receipt.fallback = 1;
            receipt.untouched_columns = 0;
            receipt.grouped_columns = 0;
            receipt.sign_folded_columns = 0;
            return Self::dense();
        }
        receipt.compiled = 1;
        receipt.input_columns = n as u64;
        receipt.active_columns = active as u64;
        view
    }

    fn try_compile(req: &DualSearchRequest<'_>, receipt: &mut SupportSearchStats) -> Option<Self> {
        let n = req.rc.len();
        let depth = req.rows.len();
        let mut groups: Vec<Group> = Vec::new();
        groups.try_reserve_exact(n).ok()?;
        let mut folded_dot = Vec::new();
        folded_dot.try_reserve_exact(depth).ok()?;
        folded_dot.resize(depth, 0.0);
        let mut delta = Vec::new();
        delta.try_reserve_exact(depth).ok()?;
        for (i, (&l0, &l1)) in req.l0.iter().zip(req.l1).enumerate() {
            if i & super::POLL_MASK == 0 && poll(req.deadline).is_err() {
                return None;
            }
            // Bounds on |(near-initial_near)-(far-initial_far)| over all
            // feasible rays. The two multiplier ceilings are independent.
            let down = add_up(l0, sub_up(l1.max(MULTIPLIER_LIMIT), l1));
            let up = add_up(sub_up(l0.max(MULTIPLIER_LIMIT), l0), l1);
            delta.push(down.max(up));
        }
        let mut hashes = HashMap::<u64, usize>::new();
        hashes.try_reserve(n.min(64)).ok()?;
        receipt.peak_bytes = receipt.peak_bytes.max(
            groups.capacity() as u64 * 16
                + hashes.capacity() as u64 * 48
                + (folded_dot.capacity() + delta.capacity()) as u64 * 8
                + 1024,
        );
        if receipt.peak_bytes > req.mem_cap_bytes {
            return None;
        }
        let mut folded_norm = 0.0;
        for j in 0..n {
            if j & super::POLL_MASK == 0 && poll(req.deadline).is_err() {
                return None;
            }
            let mut positive = hash(0xcbf29ce484222325, req.rc[j]);
            let mut negative = hash(0xcbf29ce484222325, -req.rc[j]);
            let mut radius = 0.0;
            let mut touched = false;
            for (i, row) in req.rows.iter().enumerate() {
                if i & super::POLL_MASK == 0 && poll(req.deadline).is_err() {
                    return None;
                }
                if receipt.build_reads == MAX_BUILD_READS {
                    return None;
                }
                receipt.build_reads += 1;
                let a = row.a[j];
                positive = hash(positive, a);
                negative = hash(negative, -a);
                touched |= a != 0.0;
                radius = add_up(radius, mul_up(a.abs(), delta[i]));
            }
            if receipt.build_reads > MAX_BUILD_READS {
                return None;
            }
            if !touched {
                folded_norm += req.rc[j].abs();
                receipt.untouched_columns += 1;
                continue;
            }
            // An infinite/uncertain radius refuses folding. Strict outward
            // dominance is essential; equality still has a real breakpoint.
            if req.rc[j].abs() > radius {
                let sign = if req.rc[j] > 0.0 { 1.0 } else { -1.0 };
                folded_norm += req.rc[j].abs();
                for (i, row) in req.rows.iter().enumerate() {
                    if i & super::POLL_MASK == 0 && poll(req.deadline).is_err() {
                        return None;
                    }
                    if receipt.build_reads == MAX_BUILD_READS {
                        return None;
                    }
                    receipt.build_reads += 1;
                    folded_dot[i] += sign * row.a[j];
                    if !folded_dot[i].is_finite() {
                        return None;
                    }
                }
                receipt.sign_folded_columns += 1;
                continue;
            }
            let signature = positive.min(negative);
            let mut duplicate = false;
            if let Some(&g) = hashes.get(&signature) {
                let k = groups[g].original;
                let mut same = req.rc[j] == req.rc[k];
                let mut opposite = req.rc[j] == -req.rc[k];
                for (i, row) in req.rows.iter().enumerate() {
                    if i & super::POLL_MASK == 0 && poll(req.deadline).is_err() {
                        return None;
                    }
                    if receipt.build_reads > MAX_BUILD_READS - 2 {
                        return None;
                    }
                    receipt.build_reads += 2;
                    same &= row.a[j] == row.a[k];
                    opposite &= row.a[j] == -row.a[k];
                    if !(same || opposite) {
                        break;
                    }
                }
                if same || opposite {
                    groups[g].weight += 1;
                    receipt.grouped_columns += 1;
                    duplicate = true;
                }
            }
            if !duplicate {
                let g = groups.len();
                groups.push(Group {
                    original: j,
                    weight: 1,
                });
                // A collision can only reduce compression. Never replace a
                // checked signature by a hash-equality assumption.
                if hashes.len() < MAX_HASHES && !hashes.contains_key(&signature) {
                    hashes.try_reserve(1).ok()?;
                    hashes.insert(signature, g);
                }
            }
            receipt.peak_bytes = receipt.peak_bytes.max(
                groups.capacity() as u64 * 16
                    + hashes.capacity() as u64 * 48
                    + (folded_dot.capacity() + delta.capacity()) as u64 * 8
                    + 1024,
            );
            if receipt.peak_bytes > req.mem_cap_bytes
                || poll(req.deadline).is_err()
                || receipt.build_reads > MAX_BUILD_READS
            {
                return None;
            }
        }
        if !folded_norm.is_finite() || poll(req.deadline).is_err() {
            return None;
        }
        receipt.peak_bytes = receipt.peak_bytes.max(
            groups.capacity() as u64 * 16
                + hashes.capacity() as u64 * 48
                + (folded_dot.capacity() + delta.capacity()) as u64 * 8
                + 1024,
        );
        Some(Self {
            groups: Some(groups),
            folded_dot,
            folded_norm,
        })
    }
}

/// No search vectors exist during compilation. The hash pilot holds at most
/// MAX_HASHES entries: four buckets per requested entry and 48 bytes per bucket
/// conservatively cover SwissTable's spare buckets, control bytes and alignment.
/// The actual HashMap capacity is also charged after each bounded reservation.
/// This deliberately loose admission can decline compression; dense admission
/// remains the original public working_bytes estimate.
fn compiler_admission_bytes(n: usize, depth: usize) -> u64 {
    (n as u64)
        .saturating_mul(16)
        .saturating_add((depth as u64).saturating_mul(16))
        .saturating_add((n.min(MAX_HASHES) as u64).saturating_mul(4 * 48))
        .saturating_add(4096)
}

fn hash(previous: u64, value: f64) -> u64 {
    // Signed zeros have identical support signatures.
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    (previous ^ bits).wrapping_mul(0x100000001b3)
}

fn next_up(value: f64) -> f64 {
    // All callers operate on nonnegative radii/distances (including +infinity).
    // Bit-increment rounding would be wrong for a negative finite number.
    debug_assert!(value >= 0.0);
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        f64::from_bits(1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}
fn add_up(a: f64, b: f64) -> f64 {
    next_up(a + b)
}
fn sub_up(a: f64, b: f64) -> f64 {
    next_up(a - b)
}
fn mul_up(a: f64, b: f64) -> f64 {
    next_up(a * b)
}
