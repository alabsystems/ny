// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Private heterogeneous CLOSED phase covers of the direct scalar source.
//! This module owns no imported rows, solver status or verdict. Whole signed
//! Gemm1 contraction precedes support of the frozen feature box. Every child
//! retains the parent's finite floor; installing a split always installs BOTH
//! children. Zero boundaries belong to both children. No cell is pruned.

use super::*;

const MAX_DEPTH: usize = 4;
const MAX_SPLITS: usize = 8;
const MAX_LEAVES: usize = 24;
const MAX_SEEDS: usize = 16;
const MAX_COORDINATES: usize = 16;
const MAX_ROOTS: usize = 16;
const MAX_CONTRACTION_FEATURES: usize = 65_536;

#[derive(Default)]
pub(super) struct Report {
    pub(super) leaves: Vec<usize>,
    splits: usize,
    max_depth: usize,
    seed_checks: usize,
    cache_hits: usize,
    fold_terms: usize,
    cells_checked: usize,
    search_trials: usize,
    recipient_reprices: usize,
    classes_improved: usize,
    slice_expired: bool,
    authority_expired: bool,
}

impl Report {
    pub(super) fn contraction_counts(&self) -> (usize, usize, usize) {
        (self.seed_checks, self.cache_hits, self.fold_terms)
    }
    pub(super) fn receipt(&self) {
        eprintln!("[direct-head-adaptive] splits={} leaves={} max_depth={} seed_checks={} cache_hits={} fold_terms={} cells_checked={} search_trials={} recipient_reprices={} classes_improved={} slice_expired={} authority_expired={}",
            self.splits, self.leaves.iter().sum::<usize>(), self.max_depth,
            self.seed_checks, self.cache_hits, self.fold_terms, self.cells_checked, self.search_trials, self.recipient_reprices,
            self.classes_improved, u8::from(self.slice_expired), u8::from(self.authority_expired));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Phase {
    head: usize,
    active: bool,
}

#[derive(Clone)]
struct Cell {
    path: Vec<Phase>,
    floor: f64,
}

// The frontier has no importer. Its only shape change is this atomic split;
// inherited bounds remain valid on both subsets, including an empty subset.
// Keeping all cells deliberately avoids any floating infeasibility authority.
struct Cover {
    cells: Vec<Cell>,
}

impl Cover {
    fn new(floor: f64) -> Option<Self> {
        good(floor).then_some(Self {
            cells: vec![Cell {
                path: Vec::new(),
                floor,
            }],
        })
    }

    fn split(&mut self, index: usize, head: usize) -> Option<(usize, usize)> {
        let parent = self.cells.get(index)?;
        if parent.path.len() >= MAX_DEPTH || parent.path.iter().any(|p| p.head == head) {
            return None;
        }
        let mut left = parent.clone();
        let mut right = parent.clone();
        left.path.push(Phase {
            head,
            active: false,
        });
        right.path.push(Phase { head, active: true });
        // No observable frontier exists with only the first child installed.
        self.cells[index] = left;
        let sibling = self.cells.len();
        self.cells.push(right);
        Some((index, sibling))
    }

    fn lower(&self) -> f64 {
        self.cells
            .iter()
            .map(|c| c.floor)
            .fold(f64::INFINITY, f64::min)
    }

    fn improve(&mut self, index: usize, lower: f64) {
        if good(lower) {
            self.cells[index].floor = self.cells[index].floor.max(lower);
        }
    }
}

struct Column {
    seed: Vec<f64>,
    offset: I,
}

/// Recipient-specific full-Q replay. The exact binary64 seed points need
/// not equal their authored Q/phase expression: its FULL interval residual
/// remains multiplied by full source Y, even when the point seed is zero.
fn column(
    source: &DirectScalarHeadSource,
    q: &[I],
    b: I,
    path: &[Phase],
    slopes: &[f64],
    penalties: &[f64],
    deadline: Instant,
) -> Option<Column> {
    let heads = source.y.len();
    if q.len() != heads
        || slopes.len() != heads
        || penalties.len() != path.len()
        || path.len() > MAX_DEPTH
        || !b.valid()
        || path
            .iter()
            .enumerate()
            .any(|(i, p)| p.head >= heads || path[..i].iter().any(|old| old.head == p.head))
        || penalties.iter().any(|p| !good(*p) || *p < 0.0)
    {
        return None;
    }
    let mut seed = slopes.to_vec();
    let mut offset = b;
    for j in 0..heads {
        if !live(deadline) || !q[j].valid() || !good(seed[j]) {
            return None;
        }
        if let Some((i, phase)) = path.iter().enumerate().find(|(_, p)| p.head == j) {
            let s = f64::from(u8::from(phase.active));
            let eta = 2.0 * s - 1.0;
            let nominal = 0.5 * q[j].lo + 0.5 * q[j].hi;
            let mu = nominal * s - eta * penalties[i];
            if !good(mu) {
                return None;
            }
            seed[j] = mu;
            let residual = q[j]
                .mul(I::p(s))?
                .sub(I::p(eta).mul(I::p(penalties[i]))?)?
                .sub(I::p(mu))?;
            offset = offset.add(residual.mul(source.y[j])?)?;
        } else {
            offset = offset.add(I::p(scalar_intercept(q[j], seed[j], source.y[j])?))?;
        }
    }
    Some(Column { seed, offset })
}

/// Same-source exact point-seed cache. It stores only a global lower bound
/// on sum(seed_j*y_j), never a cell/margin bound. Q/B/phase residuals are
/// reconstructed for every recipient. No interval widths are subtracted.
pub(super) struct Contractions<'a, 'n> {
    eng: &'a BackwardEngine<'n>,
    source: &'a DirectScalarHeadSource,
    completed: Vec<(Vec<u64>, f64)>,
    attempted: usize,
}

impl<'a, 'n> Contractions<'a, 'n> {
    pub(super) fn new(eng: &'a BackwardEngine<'n>, source: &'a DirectScalarHeadSource) -> Self {
        Self {
            eng,
            source,
            completed: Vec::new(),
            attempted: 0,
        }
    }

    pub(super) fn belongs_to(&self, source: &DirectScalarHeadSource) -> bool {
        std::ptr::eq(self.source, source)
            && source
                .seal
                .same_frame(self.eng, source.target, &source.adversaries)
    }

    pub(super) fn lower(
        &mut self,
        seed: &[f64],
        deadline: Instant,
        report: &mut Report,
    ) -> Option<f64> {
        if !live(deadline)
            || !fp_qualified()
            || seed.len() != self.source.y.len()
            || seed.iter().any(|v| !good(*v))
        {
            return None;
        }
        let key: Vec<_> = seed.iter().map(|v| v.to_bits()).collect();
        if let Some((_, lower)) = self.completed.iter().find(|(old, _)| *old == key) {
            report.cache_hits += 1;
            return Some(*lower);
        }
        if self.attempted >= MAX_SEEDS {
            return None;
        }
        self.attempted += 1;
        report.seed_checks += 1;
        let (w, bias, error, rel, (heads, width)) = match &self.eng.net.ops[self.eng.net.i_gemm1] {
            TwinOp::Gemm {
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
                ..
            } => (weight, bias, bias_err, *weight_rel_err, *shape),
            _ => return None,
        };
        if heads != seed.len()
            || width != self.source.features.len()
            || width > MAX_CONTRACTION_FEATURES
            || w.len() != heads.checked_mul(width)?
            || bias.len() != heads
            || error.len() != heads
            || !good(rel)
            || rel < 0.0
        {
            return None;
        }
        let mut contracted = Affine::zero(width);
        for j in 0..heads {
            if !live(deadline) {
                return None;
            }
            // Exactly zero point coefficients multiply the entire authored
            // row by zero. Any selected Q residual is outside this cache.
            if seed[j] == 0.0 {
                continue;
            }
            let point = I::p(seed[j]);
            if !good(error[j]) || error[j] < 0.0 {
                return None;
            }
            let b = I::p(bias[j]).add(I {
                lo: -error[j],
                hi: error[j],
            })?;
            contracted.b = contracted.b.add(point.mul(b)?)?;
            for k in 0..width {
                if k & 63 == 0 && !live(deadline) {
                    return None;
                }
                report.fold_terms += 1;
                let nominal = w[j * width + k];
                // UP error product is formed BEFORE multiplication by a
                // large seed/feature. Underflow cannot silently erase dust.
                let dust = I::p(rel).mul(I::p(nominal.abs()))?.hi;
                let authored = I::p(nominal).add(I {
                    lo: -dust,
                    hi: dust,
                })?;
                contracted.a[k] = contracted.a[k].add(point.mul(authored)?)?;
            }
        }
        let lower = contracted.lower(&self.source.features, deadline)?;
        if !good(lower) || !fp_qualified() || !live(deadline) {
            return None;
        }
        self.completed.push((key, lower));
        Some(lower)
    }
}

// Search-only sampled geometry. It cannot become a source enclosure or
// replace a full-width contraction. Only an entire checked point seed enters
// the cache above. Coordinate roots and scalar slope optimization are advice.
struct Geometry {
    rows: Vec<Vec<f64>>,
    bias: Vec<f64>,
    centers: Vec<f64>,
    radii: Vec<f64>,
}

impl Geometry {
    fn build(
        eng: &BackwardEngine<'_>,
        source: &DirectScalarHeadSource,
        deadline: Instant,
    ) -> Option<Self> {
        let (w, b, (heads, width)) = eng.net.gemm1();
        let n = width.min(FUNCTIONAL_MAX_FEATURES);
        if n == 0 || heads != source.y.len() {
            return None;
        }
        let features: Vec<_> = (0..n)
            .map(|k| if n == 1 { 0 } else { k * (width - 1) / (n - 1) })
            .collect();
        let mut rows = Vec::with_capacity(heads);
        for j in 0..heads {
            if !live(deadline) {
                return None;
            }
            rows.push(features.iter().map(|&k| w[j * width + k]).collect());
        }
        let centers = features
            .iter()
            .map(|&k| 0.5 * source.features[k].lo + 0.5 * source.features[k].hi)
            .collect();
        let radii = features
            .iter()
            .map(|&k| {
                (0.5 * source.features[k].hi - 0.5 * source.features[k].lo) * width as f64
                    / n as f64
            })
            .collect();
        Some(Self {
            rows,
            bias: b.to_vec(),
            centers,
            radii,
        })
    }

    fn affine(&self, seed: &[f64], deadline: Instant) -> Option<(Vec<f64>, f64)> {
        let mut a = vec![0.0; self.centers.len()];
        let mut b = 0.0;
        for (j, &s) in seed.iter().enumerate() {
            if !live(deadline) {
                return None;
            }
            b += s * self.bias[j];
            for (k, value) in a.iter_mut().enumerate() {
                *value += s * self.rows[j][k];
            }
        }
        (good(b) && a.iter().all(|v| good(*v))).then_some((a, b))
    }

    fn score(&self, a: &[f64], b: f64, offset: f64) -> Option<f64> {
        let score = offset
            + b
            + a.iter()
                .zip(&self.centers)
                .zip(&self.radii)
                .map(|((a, c), r)| a * c - a.abs() * r)
                .sum::<f64>();
        good(score).then_some(score)
    }

    fn propose(
        &self,
        source: &DirectScalarHeadSource,
        q: &[I],
        b: I,
        path: &[Phase],
        rounds: usize,
        deadline: Instant,
        report: &mut Report,
    ) -> Option<Column> {
        let initial = scalar_slopes(q, &source.y, deadline)?;
        let mut coordinate_scores = Vec::new();
        for j in (0..q.len()).filter(|&j| {
            q[j].lo > 0.0
                && source.y[j].lo < 0.0
                && source.y[j].hi > 0.0
                && !path.iter().any(|p| p.head == j)
        }) {
            if !live(deadline) {
                return None;
            }
            // Nominal duplicate rows are useful SEARCH priorities only;
            // independent fold/bias balls remain in the directed checker.
            let matches = path.iter().any(|p| self.rows[j] == self.rows[p.head]);
            let debt = (0.5 * q[j].lo + 0.5 * q[j].hi) * (source.y[j].hi - source.y[j].lo);
            coordinate_scores.push((j, matches, debt));
        }
        coordinate_scores
            .sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)).then(a.0.cmp(&b.0)));
        let coordinates: Vec<_> = coordinate_scores
            .into_iter()
            .take(MAX_COORDINATES)
            .map(|v| v.0)
            .collect();
        let mut chosen = None;
        let mut chosen_score = f64::NEG_INFINITY;
        // Independent starts preserve simultaneous cancellation ridges. They
        // have no authority; only the best finite candidate is contracted.
        for start in 0..3 {
            let mut slopes = initial.clone();
            for &j in &coordinates {
                if start != 0 {
                    slopes[j] =
                        (0.5 * q[j].lo + 0.5 * q[j].hi) * if start == 1 { 0.0 } else { 0.5 };
                }
            }
            let mut penalties = vec![0.0; path.len()];
            let mut seed = slopes.clone();
            for (i, phase) in path.iter().enumerate() {
                let s = f64::from(u8::from(phase.active));
                seed[phase.head] = (0.5 * q[phase.head].lo + 0.5 * q[phase.head].hi) * s
                    - (2.0 * s - 1.0) * penalties[i];
            }
            // SEARCH ONLY: cache individual offsets, and update one term
            // per coordinate trial. Nearest subtraction of these heuristic
            // terms is not used by the directed certificate below.
            let mut terms = Vec::with_capacity(q.len());
            for j in 0..q.len() {
                if !live(deadline) {
                    return None;
                }
                let phase = path.iter().find(|p| p.head == j).copied();
                terms.push(search_term(q[j], source.y[j], phase, 0.0, seed[j])?);
            }
            let mut offset = b.lo + terms.iter().sum::<f64>();
            let (mut a, mut intercept) = self.affine(&seed, deadline)?;
            let mut best = self.score(&a, intercept, offset)?;
            for _ in 0..rounds.min(MAX_PROPOSAL_ROUNDS) {
                let mut moved = false;
                // Coordinate IDs distinguish a scalar slope from a sign
                // multiplier. A candidate changes just one point seed.
                let ids: Vec<_> = coordinates
                    .iter()
                    .map(|&j| (j, None))
                    .chain(path.iter().enumerate().map(|(i, p)| (p.head, Some(i))))
                    .collect();
                for (j, phase) in ids {
                    if !live(deadline) {
                        return None;
                    }
                    let nominal = 0.5 * q[j].lo + 0.5 * q[j].hi;
                    let old = phase.map_or(slopes[j], |i| penalties[i]);
                    let mut values: Vec<_> = if phase.is_some() {
                        [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
                            .into_iter()
                            .map(|v| v * nominal.abs())
                            .collect()
                    } else {
                        [0.0, 0.25, 0.5, 0.75, 1.0]
                            .into_iter()
                            .map(|v| v * nominal)
                            .collect()
                    };
                    let mut indices: Vec<_> = (0..a.len()).collect();
                    indices.sort_by(|&u, &v| {
                        self.rows[j][v]
                            .abs()
                            .total_cmp(&self.rows[j][u].abs())
                            .then(u.cmp(&v))
                    });
                    for k in indices.into_iter().take(MAX_ROOTS) {
                        let denominator = self.rows[j][k];
                        if denominator == 0.0 {
                            continue;
                        }
                        let root = if let Some(i) = phase {
                            let eta = if path[i].active { 1.0 } else { -1.0 };
                            old + a[k] / (eta * denominator)
                        } else {
                            (old - a[k] / denominator).clamp(0.0, nominal)
                        };
                        if good(root) && root >= 0.0 {
                            values.push(root);
                        }
                    }
                    for value in values {
                        if !live(deadline) {
                            return None;
                        }
                        report.search_trials += 1;
                        if !good(value) || value < 0.0 {
                            continue;
                        }
                        let candidate_seed = if let Some(i) = phase {
                            let s = f64::from(u8::from(path[i].active));
                            nominal * s - (2.0 * s - 1.0) * value
                        } else {
                            value
                        };
                        if !good(candidate_seed) {
                            continue;
                        }
                        let Some(term) = search_term(
                            q[j],
                            source.y[j],
                            phase.map(|i| path[i]),
                            value,
                            candidate_seed,
                        ) else {
                            continue;
                        };
                        let trial_offset = offset - terms[j] + term;
                        let delta = candidate_seed - seed[j];
                        let trial_a: Vec<_> = a
                            .iter()
                            .zip(&self.rows[j])
                            .map(|(a, w)| a + delta * w)
                            .collect();
                        let trial_b = intercept + delta * self.bias[j];
                        if let Some(score) = self.score(&trial_a, trial_b, trial_offset) {
                            if score > best {
                                best = score;
                                if let Some(i) = phase {
                                    penalties[i] = value;
                                } else {
                                    slopes[j] = value;
                                }
                                seed[j] = candidate_seed;
                                terms[j] = term;
                                offset = trial_offset;
                                a = trial_a;
                                intercept = trial_b;
                                moved = true;
                            }
                        }
                    }
                }
                if !moved {
                    break;
                }
            }
            if best > chosen_score {
                chosen_score = best;
                chosen = Some((slopes, penalties));
            }
        }
        let (slopes, penalties) = chosen?;
        // EXACTLY ONE complete recipient repricing per returned proposal.
        // Only this directed Column can enter the whole source checker.
        report.recipient_reprices += 1;
        column(source, q, b, path, &slopes, &penalties, deadline)
    }
}

fn search_term(q: I, y: I, phase: Option<Phase>, penalty: f64, seed: f64) -> Option<f64> {
    if let Some(phase) = phase {
        let s = f64::from(u8::from(phase.active));
        let eta = 2.0 * s - 1.0;
        Some(
            q.mul(I::p(s))?
                .sub(I::p(eta).mul(I::p(penalty))?)?
                .sub(I::p(seed))?
                .mul(y)?
                .lo,
        )
    } else {
        scalar_intercept(q, seed, y)
    }
}

fn split_head(source: &DirectScalarHeadSource, q: &[I], cell: &Cell) -> Option<usize> {
    if cell.path.len() >= MAX_DEPTH || cell.floor > 0.0 {
        return None;
    }
    (0..q.len())
        .filter(|&j| {
            q[j].hi < 0.0
                && source.y[j].lo < 0.0
                && source.y[j].hi > 0.0
                && !cell.path.iter().any(|p| p.head == j)
        })
        .max_by(|&a, &b| {
            let debt = |j: usize| {
                let y = source.y[j];
                (0.5 * q[j].lo + 0.5 * q[j].hi).abs() * (-y.lo) * y.hi / (y.hi - y.lo)
            };
            debt(a).total_cmp(&debt(b)).then(b.cmp(&a))
        })
}

fn check_cell(
    cache: &mut Contractions<'_, '_>,
    geometry: &Geometry,
    q: &[I],
    b: I,
    cell: &Cell,
    rounds: usize,
    deadline: Instant,
    report: &mut Report,
) -> Option<f64> {
    let candidate = geometry.propose(cache.source, q, b, &cell.path, rounds, deadline, report)?;
    let lower = I::p(cache.lower(&candidate.seed, deadline, report)?)
        .add(candidate.offset)?
        .lo;
    if !good(lower) || !live(deadline) || !fp_qualified() {
        return None;
    }
    report.cells_checked += 1;
    Some(lower)
}

/// The caller supplies only its own freshly built scalar source and original
/// margin array. Local expiry preserves completed complete-cover floors. The
/// caller's final source/deadline guard still controls all publication.
pub(super) fn refine(
    eng: &BackwardEngine<'_>,
    source: &DirectScalarHeadSource,
    margins: &[(Vec<I>, I)],
    bounds: &mut [f64],
    policy: HeadGroupPolicy,
    deadline: Instant,
    authority: Instant,
) -> Report {
    #[allow(unused_mut)]
    let mut deadline = deadline;
    let mut report = Report {
        leaves: vec![1; bounds.len()],
        ..Report::default()
    };
    let before = bounds.to_vec();
    if margins.len() != source.adversaries.len()
        || bounds.len() != margins.len()
        || bounds.is_empty()
        || bounds.len() > MAX_LEAVES
        || bounds.iter().any(|v| !good(*v))
        || margins
            .iter()
            .any(|(q, b)| q.len() != source.y.len() || !b.valid() || q.iter().any(|v| !v.valid()))
        || !live(deadline)
        || !live(authority)
        || !fp_qualified()
        || !source
            .seal
            .same_frame(eng, source.target, &source.adversaries)
    {
        report.slice_expired = !live(deadline);
        report.authority_expired = !live(authority);
        return report;
    }
    let mut covers: Vec<_> = bounds.iter().map(|&b| Cover::new(b).unwrap()).collect();
    // No unstable negative debt => leave the existing global/native route its
    // entire remaining opportunity. Original class order is never permuted.
    let eligible: Vec<_> = covers
        .iter()
        .enumerate()
        .filter_map(|(row, cover)| {
            split_head(source, &margins[row].0, &cover.cells[0]).map(|_| row)
        })
        .collect();
    if eligible.is_empty() {
        return report;
    }
    let Some(geometry) = Geometry::build(eng, source, deadline) else {
        report.slice_expired = !live(deadline);
        return report;
    };
    let mut cache = Contractions {
        eng,
        source,
        completed: Vec::new(),
        attempted: 0,
    };
    for &row in &eligible {
        if let Some(lower) = check_cell(
            &mut cache,
            &geometry,
            &margins[row].0,
            margins[row].1,
            &covers[row].cells[0],
            policy.proposal_rounds(),
            deadline,
            &mut report,
        ) {
            covers[row].improve(0, lower);
        }
    }
    loop {
        if !live(deadline)
            || !live(authority)
            || report.splits >= MAX_SPLITS
            || covers.iter().map(|c| c.cells.len()).sum::<usize>() >= MAX_LEAVES
            || cache.attempted >= MAX_SEEDS
        {
            break;
        }
        let mut options: Vec<_> = eligible
            .iter()
            .flat_map(|&row| {
                covers[row]
                    .cells
                    .iter()
                    .enumerate()
                    .filter_map(move |(cell, c)| {
                        split_head(source, &margins[row].0, c)
                            .map(|head| (row, cell, head, c.floor, c.path.len()))
                    })
            })
            .collect();
        // Breadth before depth across original margins, then weak leaf.
        options.sort_by(|a, b| {
            a.4.cmp(&b.4)
                .then(a.3.total_cmp(&b.3))
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });
        let Some(&(row, cell, head, _, depth)) = options.first() else {
            break;
        };
        let Some((left, right)) = covers[row].split(cell, head) else {
            break;
        };
        report.splits += 1;
        report.max_depth = report.max_depth.max(depth + 1);
        for child in [left, right] {
            if let Some(lower) = check_cell(
                &mut cache,
                &geometry,
                &margins[row].0,
                margins[row].1,
                &covers[row].cells[child],
                policy.proposal_rounds(),
                deadline,
                &mut report,
            ) {
                covers[row].improve(child, lower);
            }
            #[cfg(test)]
            if expire_after_child() {
                deadline = Instant::now();
                break;
            }
        }
    }
    for (row, cover) in covers.iter().enumerate() {
        bounds[row] = bounds[row].max(cover.lower());
        report.leaves[row] = cover.cells.len();
    }
    report.classes_improved = before
        .iter()
        .zip(bounds.iter())
        .filter(|(a, b)| b > a)
        .count();
    report.slice_expired = !live(deadline);
    report.authority_expired = !live(authority);
    report
}

#[cfg(test)]
thread_local! { static STOP_AFTER_FIRST_CHILD: std::cell::Cell<u8> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
fn expire_after_child() -> bool {
    STOP_AFTER_FIRST_CHILD.with(|s| {
        if s.get() == 1 {
            s.set(2);
            true
        } else {
            false
        }
    })
}
#[cfg(test)]
fn adaptive_only_attempt_with_provider<F>(
    eng: &BackwardEngine<'_>,
    target: usize,
    adversaries: &[usize],
    policy: HeadGroupPolicy,
    deadline: Instant,
    provider: F,
) -> DirectAttempt
where
    F: FnOnce() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    attempt_portfolio(eng, target, adversaries, policy, deadline, provider, false)
}

#[cfg(test)]
#[path = "adaptive_direct_head_tests.rs"]
mod tests;
