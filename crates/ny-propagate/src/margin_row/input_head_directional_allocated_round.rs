// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Incremental completed-row publication within the existing round/call caps.
use super::*;

#[derive(Default)]
pub(super) struct Report {
    planned: usize,
    covered_negative_heads: usize,
    omitted_negative_heads: usize,
    attempted: usize,
    certified: usize,
    admitted: usize,
    completed_masks: usize,
    charged_masks: usize,
    source_replays: usize,
    search_cells: usize,
    certificate_cells: usize,
    export_cells: usize,
}

impl Report {
    pub(super) fn emit(&self) {
        eprintln!("[input-head-directional-allocated] planned_groups={} covered_negative_heads={} omitted_negative_heads={} attempted={} certified={} admitted={} completed_masks={} charged_masks={} attempted_source_replays={} search_work_cells={} certificate_work_cells={} export_work_cells={}",
            self.planned,self.covered_negative_heads,self.omitted_negative_heads,self.attempted,
            self.certified,self.admitted,self.completed_masks,self.charged_masks,self.source_replays,
            self.search_cells,self.certificate_cells,self.export_cells);
    }
}

fn terms(model: &triangle::Model<'_>) -> Option<usize> {
    model
        .request()
        .rows
        .iter()
        .try_fold(model.request().bounds.len(), |n, row| {
            n.checked_add(row.coefficients.len())
        })
}

fn initialize(source: &Source, state: &mut Portfolio<'_>) -> Option<()> {
    if state.allocated_checked {
        return Some(());
    }
    state.allocated_checked = true;
    let cells = state.budget.cells;
    let candidates = allocated::discover(source, &mut state.budget);
    if candidates.len() < 2 {
        return Some(());
    }
    // Freeze owned coefficient tuples before any row can be accepted.
    let covered = candidates.iter().map(|c| c.negative.heads.len()).sum();
    state.budget.charge(source.spec.heads)?;
    let eligible = source
        .q
        .iter()
        .zip(&source.y)
        .filter(|(q, y)| q.hi < 0.0 && y.lo < 0.0 && y.hi > 0.0)
        .count();
    state.allocated_report = Some(Report {
        planned: candidates.len(),
        covered_negative_heads: covered,
        omitted_negative_heads: eligible.saturating_sub(covered),
        search_cells: state.budget.cells - cells,
        ..Report::default()
    });
    state.allocated = candidates;
    // A multirow plan REPLACES the old whole-round anchor, never adds it free.
    state.signed_attempted = true;
    Some(())
}

pub(super) fn run<'a>(
    source: &'a Source,
    state: &mut Portfolio<'a>,
    round: &mut RoundReport,
) -> Option<triangle::Model<'a>> {
    initialize(source, state)?;
    if state.allocated_next >= state.allocated.len() {
        return None;
    }
    let limit = MAX_CUTS
        .saturating_sub(round.certified)
        .min(state.slots.checked_sub(state.accepted.len())?);
    let mut model: Option<triangle::Model<'a>> = None;
    for _ in 0..limit {
        if state.allocated_next == state.allocated.len()
            || state.report.attempted >= MAX_ATTEMPTS
            || round.candidates >= MAX_CANDIDATES
            || !live(state.budget.deadline)
        {
            break;
        }
        let prepared = (|| {
            let candidate = &state.allocated[state.allocated_next];
            let old = if let Some(model) = &model {
                terms(model)?
            } else {
                state.retained.as_ref()?.root_work_hint()?
            };
            let clone = candidate
                .positive
                .0
                .len()
                .checked_add(candidate.negative.heads.len())?;
            let added = source.spec.inputs.checked_add(clone)?;
            // Reserve actual full-prefix publication/recovery/native/full-q
            // replay credits. This is data-dependent, never a fresh quota.
            let reserve = old.checked_add(added)?.checked_mul(40)?;
            let proof =
                signed::gray_work_bound(source, &candidate.negative, &candidate.positive, 0)?;
            (state
                .budget
                .cells
                .checked_add(proof)?
                .checked_add(reserve)?
                .checked_add(clone)?
                <= MAX_CELLS)
                .then_some(clone)
        })();
        let Some(clone) = prepared else {
            break;
        };
        if state.budget.charge(clone).is_none() {
            break;
        }
        let candidate = state.allocated[state.allocated_next].clone();
        state.allocated_next += 1;
        round.candidates += 1;
        round.attempted += 1;
        state.report.attempted += 1;
        let (cells, masks) = (state.budget.cells, state.budget.masks);
        let mut attempt = signed::GrayAttempt::default();
        let cut = signed::certify_gray_protected(
            source,
            candidate.negative,
            candidate.positive,
            vec![],
            &mut state.budget,
            state.publication,
            &mut attempt,
        );
        let lower = cut.as_ref().map(|c| c.lower);
        let certified = usize::from(cut.is_some());
        round.certified += certified;
        if let Some(report) = state.allocated_report.as_mut() {
            report.attempted += 1;
            report.certified += certified;
            report.completed_masks += attempt.completed_masks;
            report.source_replays += attempt.source_replays;
            report.charged_masks += state.budget.masks - masks;
            report.certificate_cells += state.budget.cells - cells;
        }
        let mut admitted = false;
        let mut refused_append = false;
        if let Some(cut) = cut {
            if let Some(previous) = model.take() {
                state.retained = Some(previous.into_phase_prefix());
            }
            let cells = state.budget.cells;
            model = admission::admit_allocated(source, state, vec![cut]);
            admitted = model.is_some();
            if let Some(report) = &mut state.allocated_report {
                report.admitted += usize::from(admitted);
                report.export_cells += state.budget.cells - cells;
            }
            refused_append = model.is_none();
        }
        let bits = lower.map_or_else(|| "none".to_owned(), |v| format!("{:016x}", v.to_bits()));
        eprintln!("[input-head-directional-allocated-row] group={} certified={} admitted={} completed_masks={} charged_masks={} attempted_source_replays={} lower_bits={}",
            state.allocated_next,certified,usize::from(admitted),attempt.completed_masks,
            state.budget.masks-masks,attempt.source_replays,bits);
        if refused_append || !live(state.budget.deadline) {
            break;
        }
    }
    if model.is_none()
        && state
            .allocated_report
            .as_ref()
            .is_some_and(|r| r.admitted > 0)
    {
        // A refused later append did not consume or truncate the prior prefix.
        let cost = state.retained.as_ref()?.root_work_hint()?.checked_mul(8)?;
        let cells = state.budget.cells;
        if state
            .budget
            .publication_charge(cost, state.publication)
            .is_some()
        {
            model = triangle::Model::from_completed_root(
                source,
                &mut state.retained,
                state.publication.deadline,
            );
            if let Some(report) = &mut state.allocated_report {
                report.export_cells += state.budget.cells - cells;
            }
        }
    }
    model
}
