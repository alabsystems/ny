// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Private, tier-local phase relations in the frozen generator chart.
//!
//! A relation is a checked contradiction between two ORIGINAL preactivation
//! enclosures, not a proof of a margin. It proposes nonnegative sign-cut
//! multipliers only. The unchanged problem evaluator certifies every proposal;
//! no relation, scope hash or discovery score can remove a node from the cover.
//! This deliberately does not relabel generator columns as original inputs.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;

use sha2::{Digest, Sha256};

use super::super::zono_state::{tail_layers, FrozenDualFrame, FrozenDualState, ReluSite};
use super::{breakpoint::Accepted, FrozenDualProblem, NodeScratch, Side};
use crate::layers::softmax::weighted_support::Interval as I;

const TERMS: u64 = 262_144;
const ROWS: usize = 16;
const SAMPLE: usize = 32;
const PAIRS: usize = 4;
const MAX_BYTES: u64 = 256 * 1024;

mod weights;

/// Proposal receipts. Only `eval_improved`/`eval_pruned` describe bounds
/// independently certified by the unchanged frozen-dual evaluator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RelationStats {
    /// Candidate pairs inspected or node proposals attempted.
    pub attempted: u64,
    /// Strict phase conflicts admitted, or current literal matches admitted.
    pub admitted: u64,
    /// Scope, shape, resource or deadline refusals.
    pub refused: u64,
    /// Original coefficient geometry replayed during class binding.
    pub replayed: u64,
    /// Multiplier proposals actually evaluated (not eliminations).
    pub used: u64,
    /// Proposals improving the independent certified bound.
    pub eval_improved: u64,
    /// Such improvements whose independent certificate is strictly positive.
    pub eval_pruned: u64,
    /// Reserved scalar visits, including metadata inspection and original-row replay.
    pub work_reserved: u64,
    /// Largest length-only complete scope census required by a class binding.
    pub scope_census: u64,
    /// Bindings declined because that census exceeds the existing tier cap.
    pub scope_over_cap: u64,
    /// Optional sampled convex ratio searches after both old ratios failed.
    pub weight_refinements: u64,
    /// Refined ratios independently checked on every original coordinate.
    pub weight_replays: u64,
}

impl RelationStats {
    pub(super) fn merge(&mut self, other: Self) {
        self.attempted += other.attempted;
        self.admitted += other.admitted;
        self.refused += other.refused;
        self.replayed += other.replayed;
        self.used += other.used;
        self.eval_improved += other.eval_improved;
        self.eval_pruned += other.eval_pruned;
        self.work_reserved += other.work_reserved;
        self.scope_census = self.scope_census.max(other.scope_census);
        self.scope_over_cap += other.scope_over_cap;
        self.weight_refinements += other.weight_refinements;
        self.weight_replays += other.weight_replays;
    }
}

#[derive(Debug)]
struct Allowance(AtomicU64);

impl Allowance {
    fn reserve(&self, amount: u64, deadline: Option<Instant>, stats: &mut RelationStats) -> bool {
        if super::past(deadline) {
            stats.refused += 1;
            return false;
        }
        if self
            .0
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                left.checked_sub(amount)
            })
            .is_err()
        {
            stats.refused += 1;
            return false;
        }
        stats.work_reserved += amount;
        true
    }
}

#[derive(Debug, Clone)]
struct Conflict {
    rows: [usize; 2],
    hashes: [[u8; 32]; 2],
    phases: [Side; 2],
    weights: [f64; 2],
    /// Strict, outward LOWER bound on the contradiction; scheduling only.
    contradiction: f64,
}

/// Lives inside one tier, with no public import or mutation API.
#[derive(Debug, Default)]
pub(super) struct Cache {
    scope: Option<[u8; 32]>,
    conflicts: Vec<Conflict>,
    allowance: Option<Arc<Allowance>>,
    /// Terminal optional decline: subsequent classes cannot replenish work.
    retired: bool,
}

pub(super) struct ClassHints {
    conflicts: Vec<Conflict>,
    allowance: Option<Arc<Allowance>>,
    pub stats: RelationStats,
}

impl Cache {
    /// Release unused optional storage, retaining a terminal decline instead
    /// of creating a fresh allowance for a later class.
    pub fn retire(&mut self) {
        self.conflicts = Vec::new();
        self.allowance = None;
        self.retired = true;
    }

    pub fn bytes(&self) -> u64 {
        (self.conflicts.capacity() * size_of::<Conflict>()) as u64
            + self.allowance.as_ref().map_or(0, |_| {
                (size_of::<Allowance>() + 2 * size_of::<usize>()) as u64
            })
    }

    /// Only called on full certified states, original per-class or M3 shared.
    /// Any per-class change of a common chart or selected row refuses
    /// transfer; changed tail slopes are never eligible geometry.
    pub fn bind(
        &mut self,
        frame: &FrozenDualFrame,
        state: &FrozenDualState,
        order: &[usize],
        deadline: Option<Instant>,
        byte_cap: u64,
    ) -> Option<ClassHints> {
        let mut stats = RelationStats::default();
        if self.retired {
            stats.refused += 1;
            return Some(ClassHints {
                conflicts: Vec::new(),
                allowance: None,
                stats,
            });
        }
        // Census uses lengths only. No geometry is scanned before admission.
        let words =
            scope_words(frame, state).saturating_add((order.len() as u64).saturating_mul(16));
        stats.scope_census = words;
        stats.scope_over_cap = u64::from(words > TERMS);
        let scratch = (ROWS * SAMPLE * 8
            + ROWS * size_of::<usize>()
            + PAIRS * 4 * size_of::<Conflict>()
            + frame.layers.len()
            + 8192) as u64;
        if scratch + self.bytes() + (PAIRS * 4 * size_of::<Conflict>()) as u64
            > byte_cap.min(MAX_BYTES)
            || words > TERMS
        {
            stats.refused += 1;
            return Some(ClassHints {
                conflicts: Vec::new(),
                allowance: None,
                stats,
            });
        }
        let allowance = self
            .allowance
            .get_or_insert_with(|| Arc::new(Allowance(AtomicU64::new(TERMS))))
            .clone();
        if !allowance.reserve(words, deadline, &mut stats) {
            return Some(ClassHints {
                conflicts: Vec::new(),
                allowance: Some(allowance),
                stats,
            });
        }
        if !shape_ok(frame, state, order, deadline) {
            stats.refused += 1;
            return Some(ClassHints {
                conflicts: Vec::new(),
                allowance: Some(allowance),
                stats,
            });
        }
        let tail = tail_layers(frame);
        let Some(scope) = scope_hash(frame, state, &tail, deadline) else {
            stats.refused += 1;
            return Some(ClassHints {
                conflicts: Vec::new(),
                allowance: Some(allowance),
                stats,
            });
        };
        if self.scope.is_some_and(|old| old != scope) {
            stats.refused += 1;
            return Some(ClassHints {
                conflicts: Vec::new(),
                allowance: Some(allowance),
                stats,
            });
        }
        if self.scope.is_none() {
            self.scope = Some(scope);
            let eligible: Vec<_> = order
                .iter()
                .copied()
                .filter(|&row| !tail[state.layer[row]])
                .take(ROWS)
                .collect();
            discover(
                state,
                &eligible,
                &allowance,
                deadline,
                &mut stats,
                &mut self.conflicts,
            );
        }
        let mut bound = Vec::with_capacity(self.conflicts.len());
        for cached in &self.conflicts {
            // Two complete rank lookups are inspected even when a row is
            // absent. Reserve before either scan, including missing-row cases.
            if !allowance.reserve(
                (order.len() as u64).saturating_mul(2).saturating_add(8),
                deadline,
                &mut stats,
            ) {
                break;
            }
            let positions = cached.rows.map(|row| order.iter().position(|&r| r == row));
            let [Some(p0), Some(p1)] = positions else {
                continue;
            };
            if !allowance.reserve((2 * state.n_gens + 32) as u64, deadline, &mut stats) {
                break;
            }
            let Some(h0) = row_hash(state, cached.rows[0], deadline) else {
                stats.refused += 1;
                break;
            };
            let Some(h1) = row_hash(state, cached.rows[1], deadline) else {
                stats.refused += 1;
                break;
            };
            stats.replayed += 1;
            if [h0, h1] != cached.hashes {
                stats.refused += 1;
                continue;
            }
            let mut relation = cached.clone();
            relation.rows = [p0, p1];
            bound.push(relation);
        }
        Some(ClassHints {
            conflicts: bound,
            allowance: Some(allowance),
            stats,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct LevelHints<'a> {
    class: &'a ClassHints,
    n: usize,
    depth: usize,
}

impl ClassHints {
    pub fn active(&self) -> bool {
        !self.conflicts.is_empty()
    }
    pub fn bytes(&self) -> u64 {
        (self.conflicts.capacity() * size_of::<Conflict>()) as u64 + size_of::<Self>() as u64
    }
    pub fn work_bytes(depth: usize) -> u64 {
        (depth as u64).saturating_mul(4 * 8).saturating_add(1024)
    }

    pub fn admit_level(
        &self,
        n: usize,
        depth: usize,
        sides: &[u8],
        deadline: Option<Instant>,
        stats: &mut RelationStats,
    ) -> Option<LevelHints<'_>> {
        if depth == 0
            || !sides.len().is_multiple_of(depth)
            || !self
                .conflicts
                .iter()
                .any(|c| c.rows.iter().all(|&i| i < depth))
        {
            return None;
        }
        let count = sides.len() / depth;
        let allowance = self.allowance.as_ref()?;
        // Price literal inspection first, even if subsequent replay admission
        // refuses. Only matching children need the extra original evaluator.
        if !allowance.reserve(
            (count as u64)
                .saturating_mul(self.conflicts.len() as u64)
                .saturating_mul(8)
                .saturating_add(128),
            deadline,
            stats,
        ) {
            return None;
        }
        let level = LevelHints {
            class: self,
            n,
            depth,
        };
        let mut matches = 0u64;
        for (i, row) in sides.chunks_exact(depth).enumerate() {
            if i % 256 == 0 && super::past(deadline) {
                stats.refused += 1;
                return None;
            }
            matches += u64::from(level.matches_bytes(row));
        }
        if matches == 0 {
            return None;
        }
        // Two complete scalar evaluations cover M4's incumbent plus proposal;
        // scalar/block paths need only the latter. This is a visit bound,
        // not a claim about runtime or CPU instruction count.
        let per_node = (n as u64)
            .saturating_mul((depth as u64).saturating_mul(6).saturating_add(12))
            .saturating_add((depth as u64).saturating_mul(128))
            .saturating_add(256);
        if !allowance.reserve(per_node.saturating_mul(matches), deadline, stats) {
            return None;
        }
        Some(level)
    }
}

impl LevelHints<'_> {
    pub fn matches_bytes(&self, sides: &[u8]) -> bool {
        sides.len() == self.depth
            && self.class.conflicts.iter().any(|c| {
                c.rows.iter().all(|&i| i < sides.len())
                    && sides[c.rows[0]] == u8::from(c.phases[0] == Side::On)
                    && sides[c.rows[1]] == u8::from(c.phases[1] == Side::On)
            })
    }
    pub fn propose(
        &self,
        prob: &FrozenDualProblem,
        sides: &[Side],
        l0: &[f64],
        l1: &[f64],
        incumbent: f64,
        deadline: Option<Instant>,
        scratch: &mut NodeScratch,
        stats: &mut RelationStats,
    ) -> Option<Accepted> {
        if prob.n() != self.n
            || sides.len() != self.depth
            || l0.len() != sides.len()
            || l1.len() != sides.len()
            || !incumbent.is_finite()
            || super::past(deadline)
        {
            return None;
        }
        let conflict = self.class.conflicts.iter().find(|c| {
            c.rows.iter().all(|&i| i < sides.len())
                && sides[c.rows[0]] == c.phases[0]
                && sides[c.rows[1]] == c.phases[1]
        })?;
        stats.attempted += 1;
        // The WHOLE level's worst-case replay was reserved before parallel
        // workers started. Resource exhaustion cannot choose different nodes
        // according to worker arrival order; each node proposes at most once.
        stats.admitted += 1;
        let mut m0 = l0.to_vec();
        let m1 = l1.to_vec();
        let scale = (2.0 * (incumbent.abs() + 1.0) / conflict.contradiction).min(4_294_967_296.0);
        for j in 0..2 {
            m0[conflict.rows[j]] += scale * conflict.weights[j];
        }
        if m0.iter().chain(&m1).any(|x| !x.is_finite() || *x < 0.0) || super::past(deadline) {
            stats.refused += 1;
            return None;
        }
        stats.used += 1;
        let bound = prob.eval_node(sides, &m0, &m1, scratch, None);
        if super::past(deadline) || !bound.certified.is_finite() || bound.certified <= incumbent {
            return None;
        }
        stats.eval_improved += 1;
        stats.eval_pruned += u64::from(bound.prunes());
        Some(Accepted {
            bound,
            l0: m0,
            l1: m1,
        })
    }
}

fn shape_ok(
    frame: &FrozenDualFrame,
    state: &FrozenDualState,
    order: &[usize],
    deadline: Option<Instant>,
) -> bool {
    if order.len() > super::FROZEN_DUAL_MAX_DEPTH {
        return false;
    }
    let mut sorted = [usize::MAX; super::FROZEN_DUAL_MAX_DEPTH];
    sorted[..order.len()].copy_from_slice(order);
    sorted[..order.len()].sort_unstable();
    if sorted[..order.len()].windows(2).any(|w| w[0] == w[1]) {
        return false;
    }
    let s = state.num_unstable();
    for (i, &li) in state.layer.iter().enumerate() {
        if i % 256 == 0 && super::past(deadline) || li >= frame.layers.len() {
            return false;
        }
    }
    state.t == frame.t
        && state.n_gens == frame.n_gens
        && state.n_in_gens == frame.n_in_gens
        && s == frame.n_unstable
        && state.a.len() == s.saturating_mul(state.n_gens)
        && [&state.r, &state.lam, &state.mu, &state.lo, &state.hi]
            .iter()
            .all(|v| v.len() == s)
        && [&state.enc, &state.layer, &state.neuron]
            .iter()
            .all(|v| v.len() == s)
        && state.site.len() == s
        && order.len() <= super::FROZEN_DUAL_MAX_DEPTH
        && order.iter().all(|&r| r < s)
}

fn scope_words(frame: &FrozenDualFrame, state: &FrozenDualState) -> u64 {
    let flat = frame.lo.len()
        + frame.hi.len()
        + frame.mid.len()
        + frame.rad.len()
        + frame.in_cols.len()
        + frame.layer_of_op.len();
    let layers = frame.layers.iter().fold(0u64, |sum, l| {
        sum.saturating_add(
            (l.l.len() + l.u.len() + l.alpha.len() + l.s.len() + l.unst.len() + l.pos.len() + 16)
                as u64,
        )
    });
    let magnitudes = frame
        .mag
        .iter()
        .chain(&frame.wmag)
        .fold(0u64, |sum, v| sum.saturating_add(v.len() as u64 + 1));
    (flat as u64)
        .saturating_add(layers)
        .saturating_add(magnitudes)
        .saturating_add((state.num_unstable() as u64).saturating_mul(8))
        .saturating_add((frame.layers.len() as u64).saturating_mul(frame.layers.len() as u64))
        .saturating_add(128)
}

fn word(hash: &mut Sha256, value: u64) {
    hash.update(value.to_le_bytes());
}
fn site(hash: &mut Sha256, value: ReluSite) {
    match value {
        ReluSite::Head => word(hash, u64::MAX),
        ReluSite::Trunk(i) => word(hash, i as u64),
    }
}
fn floats(hash: &mut Sha256, values: &[f64], deadline: Option<Instant>) -> Option<()> {
    word(hash, values.len() as u64);
    for (i, &v) in values.iter().enumerate() {
        if i % 256 == 0 && super::past(deadline) {
            return None;
        }
        if !v.is_finite() {
            return None;
        }
        word(hash, v.to_bits());
    }
    Some(())
}
fn indices(hash: &mut Sha256, values: &[usize], deadline: Option<Instant>) -> Option<()> {
    word(hash, values.len() as u64);
    for (i, &v) in values.iter().enumerate() {
        if i % 256 == 0 && super::past(deadline) {
            return None;
        }
        word(hash, v as u64);
    }
    Some(())
}

fn scope_hash(
    frame: &FrozenDualFrame,
    state: &FrozenDualState,
    tail: &[bool],
    deadline: Option<Instant>,
) -> Option<[u8; 32]> {
    let mut h = Sha256::new();
    h.update(b"ny.fd.shared.phase-relations.v1/exact-real/f64-outward/generator-chart");
    for v in [
        frame.n_in,
        frame.n_in_gens,
        frame.n_unstable,
        frame.n_gens,
        frame.t,
        frame.n_out,
        frame.out_tensor,
        frame.max_width,
        frame.layers.len(),
    ] {
        word(&mut h, v as u64);
    }
    for v in [&frame.lo, &frame.hi, &frame.mid, &frame.rad] {
        floats(&mut h, v, deadline)?;
    }
    indices(&mut h, &frame.in_cols, deadline)?;
    indices(&mut h, &frame.layer_of_op, deadline)?;
    for l in &frame.layers {
        site(&mut h, l.site);
        for v in [l.op, l.input, l.col0] {
            word(&mut h, v as u64);
        }
        for v in [&l.l, &l.u, &l.alpha, &l.s] {
            floats(&mut h, v, deadline)?;
        }
        indices(&mut h, &l.unst, deadline)?;
        indices(&mut h, &l.pos, deadline)?;
    }
    for plane in [&frame.mag, &frame.wmag] {
        word(&mut h, plane.len() as u64);
        for v in plane {
            floats(&mut h, v, deadline)?;
        }
    }
    // Every slope read by rows is part of the common chart. The M3 builder
    // independently forbids reading the excluded class-dependent tail.
    for k in 0..state.num_unstable() {
        if k % 256 == 0 && super::past(deadline) {
            return None;
        }
        if !tail[state.layer[k]] {
            if !(0.0..=1.0).contains(&state.lam[k]) || state.mu[k] <= 0.0 {
                return None;
            }
            for v in [state.lam[k], state.mu[k]] {
                if !v.is_finite() {
                    return None;
                }
                word(&mut h, v.to_bits());
            }
            for v in [k, state.enc[k], state.layer[k], state.neuron[k]] {
                word(&mut h, v as u64);
            }
            site(&mut h, state.site[k]);
        }
    }
    Some(h.finalize().into())
}

fn row_hash(state: &FrozenDualState, row: usize, deadline: Option<Instant>) -> Option<[u8; 32]> {
    let mut h = Sha256::new();
    h.update(b"ny.fd.original-preactivation-row.v1");
    for v in [
        row,
        state.n_gens,
        state.enc[row],
        state.layer[row],
        state.neuron[row],
    ] {
        word(&mut h, v as u64);
    }
    site(&mut h, state.site[row]);
    floats(
        &mut h,
        &[
            state.c_in[row],
            state.r[row],
            state.lam[row],
            state.mu[row],
            state.lo[row],
            state.hi[row],
        ],
        deadline,
    )?;
    if state.r[row] < 0.0
        || !(0.0..=1.0).contains(&state.lam[row])
        || state.mu[row] <= 0.0
        || state.lo[row] > state.hi[row]
        || state.enc[row] >= state.n_gens
    {
        return None;
    }
    floats(&mut h, state.row(row), deadline)?;
    Some(h.finalize().into())
}

fn discover(
    state: &FrozenDualState,
    rows: &[usize],
    allowance: &Allowance,
    deadline: Option<Instant>,
    stats: &mut RelationStats,
    out: &mut Vec<Conflict>,
) {
    let samples = SAMPLE.min(state.n_gens);
    let selection_work = (rows.len() * samples + rows.len() * rows.len() * samples * 8) as u64;
    if !allowance.reserve(selection_work, deadline, stats) || samples == 0 {
        return;
    }
    let mut sample = [[0.0; SAMPLE]; ROWS];
    for (j, &r) in rows.iter().enumerate() {
        for i in 0..samples {
            if super::past(deadline) {
                return;
            }
            sample[j][i] = state.row(r)[i * state.n_gens / samples];
        }
    }
    // Bounded sample search proposes ratios only; original coefficients are
    // independently checked below. Scores have no certificate authority.
    let mut pairs: Vec<(f64, usize, usize, f64)> = Vec::with_capacity(PAIRS);
    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            if super::past(deadline) {
                return;
            }
            let (mut ai, mut aj, mut same, mut opposite) = (0.0, 0.0, 0.0, 0.0);
            for k in 0..samples {
                ai += sample[i][k].abs();
                aj += sample[j][k].abs();
            }
            let weight = if (ai + aj).is_finite() && ai + aj > 0.0 {
                (aj / (ai + aj)).clamp(0.0625, 0.9375)
            } else {
                0.5
            };
            for k in 0..samples {
                let a = weight * sample[i][k];
                let b = (1.0 - weight) * sample[j][k];
                same += (a + b).abs();
                opposite += (a - b).abs();
            }
            let score = same.min(opposite) / (ai + aj).max(f64::MIN_POSITIVE);
            if !score.is_finite() {
                continue;
            }
            if pairs.len() < PAIRS {
                pairs.push((score, i, j, weight));
            } else if let Some((worst, _)) = pairs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.0.total_cmp(&b.0))
            {
                if score < pairs[worst].0 {
                    pairs[worst] = (score, i, j, weight);
                }
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    // At most four failed original pairs; fixed stack storage fits the old
    // 8192-byte metadata slack. Optional rescue never interrupts old discovery.
    let mut failed = [None; PAIRS];
    let mut failed_count = 0;
    for (_, i, j, weight) in pairs {
        stats.attempted += 1;
        let ids = [rows[i], rows[j]];
        if !allowance.reserve(
            (state.n_gens as u64).saturating_mul(6).saturating_add(256),
            deadline,
            stats,
        ) {
            break;
        }
        let Some(h0) = row_hash(state, ids[0], deadline) else {
            stats.refused += 1;
            continue;
        };
        let Some(h1) = row_hash(state, ids[1], deadline) else {
            stats.refused += 1;
            continue;
        };
        let before = out.len();
        for w in [weight, 0.5] {
            if w == 0.5 && weight == 0.5 && out.iter().any(|c| c.rows == ids) {
                continue;
            }
            if let Some(found) = check_pair(state, ids, [w, 1.0 - w], deadline) {
                for (phases, contradiction) in found {
                    if out.iter().any(|c| c.rows == ids && c.phases == phases) {
                        continue;
                    }
                    stats.admitted += 1;
                    out.push(Conflict {
                        rows: ids,
                        hashes: [h0, h1],
                        phases,
                        weights: [w, 1.0 - w],
                        contradiction,
                    });
                }
            }
        }
        if out.len() == before {
            failed[failed_count] = Some((i, j, weight, [h0, h1]));
            failed_count += 1;
        }
    }
    // Preserve all old conflicts AND their subsequent original class-binding
    // credits. Rescue runs only after every old selected pair and only when
    // the baseline has no conflict to bind. This targets the actual zero-
    // engagement charts without spending useful old proposal authority.
    if !out.is_empty() {
        return;
    }
    for &(i, j, weight, hashes) in failed[..failed_count].iter().flatten() {
        let ids = [rows[i], rows[j]];
        if allowance.reserve(weights::WORK, deadline, stats) {
            stats.weight_refinements += 1;
            let centers = ids.map(|r| state.c_in[r]);
            let radii = ids.map(|r| state.r[r]);
            if let Some(w) = weights::propose(
                &sample[i][..samples],
                &sample[j][..samples],
                centers,
                radii,
                state.n_gens,
                deadline,
            ) {
                if w != weight
                    && w != 0.5
                    && allowance.reserve(
                        (state.n_gens as u64).saturating_mul(2).saturating_add(128),
                        deadline,
                        stats,
                    )
                {
                    stats.weight_replays += 1;
                    if let Some(found) = check_pair(state, ids, [w, 1.0 - w], deadline) {
                        for (phases, contradiction) in found {
                            stats.admitted += 1;
                            out.push(Conflict {
                                rows: ids,
                                hashes,
                                phases,
                                weights: [w, 1.0 - w],
                                contradiction,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// All four phase combinations are considered. No assertion is made about
/// unresolved combinations; z=0 belongs to both sides, and ties stay open.
fn check_pair(
    state: &FrozenDualState,
    rows: [usize; 2],
    weights: [f64; 2],
    deadline: Option<Instant>,
) -> Option<Vec<([Side; 2], f64)>> {
    if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return None;
    }
    let (mut plus, mut minus) = (I::point(0.0), I::point(0.0));
    for (k, (&a, &b)) in state
        .row(rows[0])
        .iter()
        .zip(state.row(rows[1]))
        .enumerate()
    {
        if k % 256 == 0 && super::past(deadline) {
            return None;
        }
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let a = I::point(a).mul(I::point(weights[0]));
        let b = I::point(b).mul(I::point(weights[1]));
        plus = plus.add(I::point(a.add(b).abs_max()));
        minus = minus.add(I::point(a.sub(b).abs_max()));
    }
    let mut result = Vec::with_capacity(4);
    for s0 in [Side::Off, Side::On] {
        for s1 in [Side::Off, Side::On] {
            let mut rhs = I::point(0.0);
            for (j, side) in [s0, s1].into_iter().enumerate() {
                let k = rows[j];
                if !state.c_in[k].is_finite() || !state.r[k].is_finite() || state.r[k] < 0.0 {
                    return None;
                }
                let center = I::point(state.c_in[k]);
                let b = (if side == Side::Off {
                    center.neg()
                } else {
                    center
                })
                .add(I::point(state.r[k]));
                rhs = rhs.add(I::point(weights[j]).mul(b));
            }
            let contradiction = (if s0 == s1 { plus } else { minus }).neg().sub(rhs);
            if contradiction.finite() && contradiction.lo > 0.0 {
                result.push(([s0, s1], contradiction.lo));
            }
        }
    }
    Some(result)
}

#[cfg(test)]
#[path = "relations_tests.rs"]
mod tests;
