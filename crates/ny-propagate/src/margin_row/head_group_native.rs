// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Optional whole-functional replay of an already complete private cover.
//!
//! For each closed phase, subtract its nonnegative sign penalties BEFORE
//! sending the whole head functional through Gemm1 and the trunk. Native
//! coefficients are exact binary64 seed points, not claimed exact authored
//! differences. Every unselected scalar intercept and full selected residual
//! interval repairs that distinction. Untouched masks retain old certificates.

use super::*;

#[derive(Default)]
pub(super) struct NativeRefinement {
    pub passes: usize,
    pub columns: usize,
    pub improved_phases: usize,
}

#[derive(Clone)]
struct Column {
    mask: usize,
    coefficients: Vec<f64>,
    offset: I,
    residual: I,
}

// A zero exact seed functional needs no numerical propagation. Its source
// residual is still charged, so rounded cancellation alone never authorizes
// this shortcut. This also avoids inventing error dust for exact zero dots.
fn zero_functional_lower(column: &Column, deadline: Instant) -> Option<f64> {
    if !live(deadline) || column.coefficients.iter().any(|v| *v != 0.0) {
        return None;
    }
    let lower = column.offset.add(column.residual)?.lo;
    (good(lower) && fp_qualified() && live(deadline)).then_some(lower)
}

pub(super) fn source_remainder(
    problem: &SourceHeadProblem,
    group: &[usize],
    deadline: Instant,
) -> Option<(Vec<f64>, I)> {
    let mut seed = vec![0.0; problem.head.len()];
    let mut offset = problem.bias;
    for (j, e) in problem.head.iter().enumerate() {
        if !live(deadline) {
            return None;
        }
        if !group.contains(&j) {
            let (v, k) = scalar_line(problem.q[j], e)?;
            seed[j] = v;
            offset = offset.add(I::p(k))?;
        }
    }
    Some((seed, offset))
}

fn source_column(
    problem: &SourceHeadProblem,
    group: &[usize],
    proposal: &Proposal,
    remainder: &[f64],
    offset: I,
) -> Option<Column> {
    if proposal.mask >= 1 << group.len()
        || proposal.phase.len() != group.len()
        || proposal.phase.iter().any(|p| !good(*p) || *p < 0.0)
        || remainder.len() != problem.head.len()
    {
        return None;
    }
    let mut coefficients = remainder.to_vec();
    let mut residual = I::p(0.0);
    for (i, &j) in group.iter().enumerate() {
        let s = f64::from(u8::from(proposal.mask & (1 << i) != 0));
        let eta = 2.0 * s - 1.0;
        let nominal = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
        let mu = nominal * s - eta * proposal.phase[i];
        if !good(mu) {
            return None;
        }
        coefficients[j] = mu;
        let error = problem.q[j]
            .mul(I::p(s))?
            .sub(I::p(eta).mul(I::p(proposal.phase[i]))?)?
            .sub(I::p(mu))?;
        residual = residual.add(error.mul(problem.head[j].y)?)?;
    }
    coefficients.iter().all(|c| good(*c)).then_some(Column {
        mask: proposal.mask,
        coefficients,
        offset,
        residual,
    })
}

// Advice only: measure cancellation in a fixed, source-owned sample of the
// first head's input coefficients. The native checker later propagates every
// actual feature and charges every fold/error term. This score proves nothing.
struct GemmAdvice {
    base: Vec<f64>,
    selected: Vec<Vec<f64>>,
    base_bias: f64,
    selected_bias: Vec<f64>,
    centers: Vec<f64>,
    radii: Vec<f64>,
    sample_scale: f64,
}

impl GemmAdvice {
    fn from_source(
        eng: &BackwardEngine<'_>,
        group: &[usize],
        remainder: &[f64],
        deadline: Instant,
    ) -> Option<Self> {
        let (weights, biases, (head, width)) = eng.net.gemm1();
        if head != remainder.len()
            || width == 0
            || weights.len() != head.checked_mul(width)?
            || group.iter().any(|j| *j >= head)
        {
            return None;
        }
        let samples = width.min(MAX_ADVICE_FEATURES);
        let features: Vec<_> = (0..samples)
            .map(|j| {
                if samples == 1 {
                    Some(0)
                } else {
                    j.checked_mul(width - 1).map(|n| n / (samples - 1))
                }
            })
            .collect::<Option<_>>()?;
        let mut base = vec![0.0; samples];
        let mut selected = vec![vec![0.0; samples]; group.len()];
        let base_bias: f64 = remainder.iter().zip(biases).map(|(v, b)| v * b).sum();
        let selected_bias: Vec<_> = group.iter().map(|&j| biases[j]).collect();
        // Advice geometry for the final trunk ReLU (through any Flatten).
        // Other tensor producers retain the old unit-box surrogate. None of
        // these sampled boxes or nearest sums is certificate authority.
        let mut tensor = match &eng.net.ops[eng.net.i_gemm1] {
            TwinOp::Gemm { input, .. } => *input,
            _ => return None,
        };
        while tensor > 0 {
            if !live(deadline) {
                return None;
            }
            match &eng.net.ops[tensor - 1] {
                TwinOp::Flatten { input } => tensor = *input,
                _ => break,
            }
        }
        let last = tensor.checked_sub(1).and_then(|op| {
            eng.root
                .layers
                .iter()
                .find(|layer| layer.op == op && layer.n == width)
        });
        let centers: Vec<_> = features
            .iter()
            .map(|&k| {
                last.map_or(0.0, |layer| {
                    0.5 * layer.l[k].max(0.0) + 0.5 * layer.u[k].max(0.0)
                })
            })
            .collect();
        let radii: Vec<_> = features
            .iter()
            .map(|&k| {
                last.map_or(1.0, |layer| {
                    0.5 * layer.u[k].max(0.0) - 0.5 * layer.l[k].max(0.0)
                })
            })
            .collect();
        for (k, &feature) in features.iter().enumerate() {
            if !live(deadline) {
                return None;
            }
            for (j, &s) in remainder.iter().enumerate() {
                base[k] += s * weights[j * width + feature];
            }
            for (i, &j) in group.iter().enumerate() {
                selected[i][k] = weights[j * width + feature];
            }
            if !good(base[k]) || selected.iter().any(|a| !good(a[k])) {
                return None;
            }
        }
        (good(base_bias)
            && selected_bias
                .iter()
                .chain(&centers)
                .chain(&radii)
                .all(|v| good(*v)))
        .then_some(Self {
            base,
            selected,
            base_bias,
            selected_bias,
            centers,
            radii,
            sample_scale: width as f64 / samples as f64,
        })
    }

    fn coefficients(
        &self,
        problem: &SourceHeadProblem,
        group: &[usize],
        p: &Proposal,
        deadline: Instant,
    ) -> Option<Vec<f64>> {
        let mut output = self.base.clone();
        for (i, &j) in group.iter().enumerate() {
            let s = f64::from(u8::from(p.mask & (1 << i) != 0));
            let eta = 2.0 * s - 1.0;
            let q = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
            let mu = q * s - eta * p.phase[i];
            if !good(mu) || !live(deadline) {
                return None;
            }
            for (k, a) in output.iter_mut().enumerate() {
                *a += mu * self.selected[i][k];
            }
        }
        output.iter().all(|a| good(*a)).then_some(output)
    }

    fn score(
        &self,
        problem: &SourceHeadProblem,
        group: &[usize],
        p: &Proposal,
        deadline: Instant,
    ) -> Option<f64> {
        let mut value = self.base_bias;
        for (i, &j) in group.iter().enumerate() {
            let s = f64::from(u8::from(p.mask & (1 << i) != 0));
            let eta = 2.0 * s - 1.0;
            let q = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
            value += (q * s - eta * p.phase[i]) * self.selected_bias[i];
        }
        let sampled: f64 = self
            .coefficients(problem, group, p, deadline)?
            .iter()
            .zip(&self.centers)
            .zip(&self.radii)
            .map(|((a, center), radius)| a * center - a.abs() * radius)
            .sum();
        value += self.sample_scale * sampled;
        (good(value) && live(deadline)).then_some(value)
    }

    fn propose(
        &self,
        problem: &SourceHeadProblem,
        group: &[usize],
        old: &Proposal,
        deadline: Instant,
    ) -> Option<Proposal> {
        let zero = Proposal {
            mask: old.mask,
            phase: vec![0.0; group.len()],
        };
        // A piecewise-linear score can have a cancellation ridge where each
        // single-coordinate move is worse even though a joint move improves
        // its bias. Preserve both starts instead of committing to the best
        // initial scalar score. This remains bounded advice, not an optimum.
        let a = self.propose_from(problem, group, zero, deadline)?;
        let b = self.propose_from(problem, group, old.clone(), deadline)?;
        if self.score(problem, group, &a, deadline)? >= self.score(problem, group, &b, deadline)? {
            Some(a)
        } else {
            Some(b)
        }
    }

    fn propose_from(
        &self,
        problem: &SourceHeadProblem,
        group: &[usize],
        mut p: Proposal,
        deadline: Instant,
    ) -> Option<Proposal> {
        let mut best = self.score(problem, group, &p, deadline)?;
        // Coordinate roots offer continuous cancellation values as well as
        // the old finite grid. They remain untrusted finite proposals; all
        // roundoff, omitted sampled features and residuals are replayed below.
        for _ in 0..MAX_PROPOSAL_ROUNDS {
            let mut moved = false;
            for (i, &j) in group.iter().enumerate() {
                let original = p.phase[i];
                let mut chosen = original;
                let eta = if p.mask & (1 << i) != 0 { 1.0 } else { -1.0 };
                let q = (0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi).abs();
                let mut values: Vec<_> = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
                    .into_iter()
                    .map(|scale| q * scale)
                    .collect();
                let coefficients = self.coefficients(problem, group, &p, deadline)?;
                let mut indices: Vec<_> = (0..self.base.len()).collect();
                indices.sort_by(|&a, &b| {
                    self.selected[i][b]
                        .abs()
                        .total_cmp(&self.selected[i][a].abs())
                        .then(a.cmp(&b))
                });
                for k in indices.into_iter().take(16) {
                    let denominator = eta * self.selected[i][k];
                    if denominator != 0.0 {
                        let root = original + coefficients[k] / denominator;
                        if good(root) && root >= 0.0 {
                            values.push(root);
                        }
                    }
                }
                for value in values {
                    if !live(deadline) {
                        return None;
                    }
                    p.phase[i] = value;
                    if let Some(score) = self.score(problem, group, &p, deadline) {
                        if score > best {
                            best = score;
                            chosen = value;
                            moved = true;
                        }
                    }
                }
                p.phase[i] = chosen;
            }
            if !moved {
                break;
            }
        }
        Some(p)
    }
}

fn replay_column(
    problem: &SourceHeadProblem,
    pass: &PassOut,
    columns: usize,
    index: usize,
    column: &Column,
    deadline: Instant,
) -> Option<f64> {
    let row = owned_row(pass, &problem.root, columns, index, true, deadline)?;
    let lower = I::p(row.lower(&problem.root, deadline)?)
        .add(column.offset)?
        .add(column.residual)?
        .lo;
    (good(lower) && fp_qualified() && live(deadline)).then_some(lower)
}

/// Refine a subset of masks only after all masks have owned certificates.
/// This function accepts no imported rows, seed, phase-bound or proposal.
pub(super) fn refine_complete_cover(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    cover: &mut RootGroupBound<'_>,
    deadline: Instant,
) -> Option<NativeRefinement> {
    let mut report = NativeRefinement::default();
    if !cover.applies_to(problem)
        || cover.group != group
        || cover.phases != 1 << group.len()
        || cover.phase_lower.len() != cover.phases
        || cover.proposals.len() != cover.phases
        || cover.phase_lower.iter().any(|v| !good(*v))
        || !fp_qualified()
        || !pack.owned_for_head_groups(eng)
    {
        return None;
    }
    if !live(deadline) || cover.lower > 0.0 {
        return Some(report);
    }
    let Some((remainder, offset)) = source_remainder(problem, group, deadline) else {
        return Some(report);
    };
    let advice = GemmAdvice::from_source(eng, group, &remainder, deadline);
    let mut attempted: Vec<Vec<Vec<f64>>> = vec![Vec::new(); cover.phases];
    let before = cover.phase_lower.clone();
    for _ in 0..MAX_NATIVE_BATCHES {
        if !live(deadline) || cover.lower > 0.0 {
            break;
        }
        let mut masks: Vec<_> = (0..cover.phases).collect();
        masks.sort_by(|&a, &b| {
            cover.phase_lower[a]
                .total_cmp(&cover.phase_lower[b])
                .then(a.cmp(&b))
        });
        let mut columns = Vec::with_capacity(MAX_NATIVE_COLUMNS);
        'masks: for mask in masks {
            if cover.phase_lower[mask] > 0.0 || !live(deadline) {
                continue;
            }
            let old = &cover.proposals[mask];
            let mut portfolio = vec![
                old.clone(),
                Proposal {
                    mask,
                    phase: vec![0.0; group.len()],
                },
            ];
            if let Some(candidate) = advice
                .as_ref()
                .and_then(|a| a.propose(problem, group, old, deadline))
            {
                portfolio.push(candidate);
            }
            for proposal in portfolio {
                if attempted[mask].contains(&proposal.phase) {
                    continue;
                }
                attempted[mask].push(proposal.phase.clone());
                if let Some(column) = source_column(problem, group, &proposal, &remainder, offset) {
                    if let Some(lower) = zero_functional_lower(&column, deadline) {
                        cover.phase_lower[mask] = cover.phase_lower[mask].max(lower);
                    } else {
                        columns.push(column);
                    }
                }
                if columns.len() == MAX_NATIVE_COLUMNS {
                    break 'masks;
                }
            }
        }
        cover.lower = cover
            .phase_lower
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if columns.is_empty() || !live(deadline) {
            break;
        }
        let mut coefficients = ndarray::Array2::zeros((problem.head.len(), columns.len()));
        for (r, column) in columns.iter().enumerate() {
            for (j, &value) in column.coefficients.iter().enumerate() {
                coefficients[[j, r]] = value;
            }
        }
        // Stored values are exact seed POINTS. Their derivation was rounded;
        // source_column retains that complete selected interval residual.
        let seed = Seed {
            s: coefficients,
            e: None,
        };
        let Ok(pass) = eng.run_with_deadline(&seed, LaneDir::Lower, deadline) else {
            break;
        };
        if !fp_qualified() || !pack.owned_for_head_groups(eng) {
            return None;
        }
        report.passes += 1;
        report.columns += columns.len();
        for (r, column) in columns.iter().enumerate() {
            if let Some(lower) = replay_column(problem, &pass, columns.len(), r, column, deadline) {
                cover.phase_lower[column.mask] = cover.phase_lower[column.mask].max(lower);
            }
        }
        cover.lower = cover
            .phase_lower
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
    }
    report.improved_phases = before
        .iter()
        .zip(&cover.phase_lower)
        .filter(|(a, b)| b > a)
        .count();
    (fp_qualified() && cover.phase_lower.iter().all(|v| good(*v))).then_some(report)
}

#[cfg(test)]
pub(super) fn replay_one_for_test(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    proposal: &Proposal,
    deadline: Instant,
) -> Option<f64> {
    replay_proposals(
        eng,
        pack,
        problem,
        group,
        std::slice::from_ref(proposal),
        deadline,
    )?
    .bounds
    .into_iter()
    .find_map(|(index, lower)| (index == 0).then_some(lower))
}

pub(super) struct ProposalReplay {
    pub bounds: Vec<(usize, f64)>,
    pub passes: usize,
    pub columns: usize,
}

/// Replay at most four externally advised multiplier alternatives. Their
/// source columns and all residuals are built here from the private source;
/// no imported row, returned LP bound or nominal cancellation grants validity.
pub(super) fn replay_proposals(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    proposals: &[Proposal],
    deadline: Instant,
) -> Option<ProposalReplay> {
    if !pack.owned_for_head_groups(eng) || !fp_qualified() || !live(deadline) {
        return None;
    }
    if proposals.len() > MAX_NATIVE_COLUMNS
        || !(2..=MAX_GROUP).contains(&group.len())
        || group
            .iter()
            .enumerate()
            .any(|(i, &j)| j >= problem.head.len() || group[..i].contains(&j))
    {
        return None;
    }
    let (remainder, offset) = source_remainder(problem, group, deadline)?;
    let mut columns = Vec::with_capacity(proposals.len());
    for (index, proposal) in proposals.iter().enumerate() {
        let column = source_column(problem, group, proposal, &remainder, offset)?;
        columns.push((index, column));
    }
    replay_columns(eng, pack, problem, columns, deadline)
}

/// Arbitrary all-head points are proposals. Every unselected scalar intercept
/// and every selected full-Q residual is rebuilt from the private source.
pub(super) fn replay_functionals(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    proposals: &[FunctionalProposal],
    deadline: Instant,
) -> Option<ProposalReplay> {
    let prepared = prepare_functionals(eng, pack, problem, group, proposals, deadline)?;
    replay_prepared_functionals(eng, pack, problem, &prepared, deadline)
}

/// Opaque source-priced columns. Cross-margin batching cannot import a seed,
/// intercept or residual; prepare_functionals alone constructs this object.
pub(super) struct PreparedFunctional {
    column: Column,
    root: Vec<I>,
    source: super::super::head_group_source::YPackSource,
}

pub(super) fn prepare_functionals(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    proposals: &[FunctionalProposal],
    deadline: Instant,
) -> Option<Vec<PreparedFunctional>> {
    problem.validate()?;
    if !fp_qualified()
        || !live(deadline)
        || proposals.len() > MAX_NATIVE_COLUMNS
        || group.len() > MAX_GROUP
        || group
            .iter()
            .enumerate()
            .any(|(i, &j)| j >= problem.head.len() || group[..i].contains(&j))
    {
        return None;
    }
    let source = pack.head_source_snapshot(eng)?;
    let mut columns = Vec::with_capacity(proposals.len());
    for p in proposals {
        if p.mask >= 1 << group.len()
            || p.slopes.len() != problem.head.len()
            || p.slopes.iter().any(|v| !good(*v))
            || p.phase.len() != group.len()
            || p.phase.iter().any(|v| !good(*v) || *v < 0.0)
        {
            return None;
        }
        let mut coefficients = p.slopes.clone();
        let mut offset = problem.bias;
        let mut residual = I::p(0.0);
        for (j, e) in problem.head.iter().enumerate() {
            if !live(deadline) {
                return None;
            }
            if let Some(i) = group.iter().position(|&g| g == j) {
                let s = f64::from(u8::from(p.mask & (1 << i) != 0));
                let eta = 2.0 * s - 1.0;
                let nominal = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
                let mu = nominal * s - eta * p.phase[i];
                if !good(mu) {
                    return None;
                }
                coefficients[j] = mu;
                let delta = problem.q[j]
                    .mul(I::p(s))?
                    .sub(I::p(eta).mul(I::p(p.phase[i]))?)?
                    .sub(I::p(mu))?;
                residual = residual.add(delta.mul(e.y)?)?;
            } else {
                let k = scalar_intercept(problem.q[j], coefficients[j], e.y)?;
                offset = offset.add(I::p(k))?;
            }
        }
        columns.push(PreparedFunctional {
            column: Column {
                mask: p.mask,
                coefficients,
                offset,
                residual,
            },
            root: problem.root.clone(),
            source: source.clone(),
        });
    }
    Some(columns)
}

pub(super) fn replay_prepared_functionals(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    prepared: &[PreparedFunctional],
    deadline: Instant,
) -> Option<ProposalReplay> {
    let references: Vec<_> = prepared.iter().collect();
    replay_prepared_functional_refs(eng, pack, problem, &references, deadline)
}

pub(super) fn replay_prepared_functional_refs(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    prepared: &[&PreparedFunctional],
    deadline: Instant,
) -> Option<ProposalReplay> {
    if prepared.len() > MAX_NATIVE_COLUMNS || !fp_qualified() || !live(deadline) {
        return None;
    }
    // Validate once per batch. Every prepared column carries a private
    // sealed snapshot whose equality check does not rehash shared source.
    let source = pack.head_source_snapshot(eng)?;
    let mut columns = Vec::with_capacity(prepared.len());
    for (index, p) in prepared.iter().enumerate() {
        if p.root.len() != problem.root.len() || !p.source.same_snapshot(&source) {
            return None;
        }
        for (k, (a, b)) in p.root.iter().zip(&problem.root).enumerate() {
            if k & 63 == 0 && !live(deadline) {
                return None;
            }
            if a.lo.to_bits() != b.lo.to_bits() || a.hi.to_bits() != b.hi.to_bits() {
                return None;
            }
        }
        columns.push((index, p.column.clone()));
    }
    replay_columns(eng, pack, problem, columns, deadline)
}

/// Search rank only: omitted features and nearest arithmetic cannot certify
/// a bound. The selected candidate is later replayed on EVERY real feature.
pub(super) fn score_functional(
    eng: &BackwardEngine<'_>,
    problem: &SourceHeadProblem,
    slopes: &[f64],
    deadline: Instant,
) -> Option<f64> {
    let advice = GemmAdvice::from_source(eng, &[], slopes, deadline)?;
    let mut score = advice.score(
        problem,
        &[],
        &Proposal {
            mask: 0,
            phase: vec![],
        },
        deadline,
    )? + problem.bias.lo;
    for (j, e) in problem.head.iter().enumerate() {
        if !live(deadline) {
            return None;
        }
        score += scalar_intercept(problem.q[j], slopes[j], e.y)?;
    }
    (good(score) && live(deadline)).then_some(score)
}

fn replay_columns(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    source_columns: Vec<(usize, Column)>,
    deadline: Instant,
) -> Option<ProposalReplay> {
    let mut report = ProposalReplay {
        bounds: Vec::with_capacity(source_columns.len()),
        passes: 0,
        columns: 0,
    };
    let mut columns = Vec::with_capacity(source_columns.len());
    for (index, column) in source_columns {
        if let Some(lower) = zero_functional_lower(&column, deadline) {
            report.bounds.push((index, lower));
        } else {
            columns.push((index, column));
        }
    }
    if !columns.is_empty() && live(deadline) {
        let mut coefficients = ndarray::Array2::zeros((problem.head.len(), columns.len()));
        for (r, (_, column)) in columns.iter().enumerate() {
            for (j, &value) in column.coefficients.iter().enumerate() {
                coefficients[[j, r]] = value;
            }
        }
        if let Ok(pass) = eng.run_with_deadline(
            &Seed {
                s: coefficients,
                e: None,
            },
            LaneDir::Lower,
            deadline,
        ) {
            if !fp_qualified() {
                return None;
            }
            report.passes += 1;
            report.columns += columns.len();
            for (r, (index, column)) in columns.iter().enumerate() {
                if let Some(lower) =
                    replay_column(problem, &pass, columns.len(), r, column, deadline)
                {
                    report.bounds.push((*index, lower));
                }
            }
        }
    }
    if !pack.owned_for_head_groups(eng) || !fp_qualified() {
        return None;
    }
    Some(report)
}
