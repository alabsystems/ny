// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded feedback changes search, never the independent geometry proof.
use super::*;
#[path = "input_head_directional_admission.rs"]
mod admission;
#[path = "input_head_directional_allocated_round.rs"]
mod allocated_round;

#[path = "input_head_directional_debt.rs"]
mod debt;
pub(super) use debt::geometry_debt;
struct Portfolio<'a> {
    retained: Option<triangle::PhasePrefix<'a>>,
    accepted: Vec<CertifiedDirection<'a>>,
    seen: Vec<Vec<u64>>,
    debt: Vec<f64>,
    gradient: Vec<f64>,
    base_gradient: Vec<f64>,
    native_track: Option<(Vec<f64>, Vec<f64>)>,
    ay_track_valid: bool,
    budget: Budget,
    report: Refinement,
    slots: usize,
    search_bound: f64,
    signed_attempted: bool,
    signed_report: Option<signed::AttemptReport>,
    publication: PublicationAuthority,
    allocated_checked: bool,
    allocated: Vec<allocated::Candidate>,
    allocated_next: usize,
    allocated_report: Option<allocated_round::Report>,
}

fn candidates(
    source: &Source,
    state: &mut Portfolio<'_>,
) -> Option<(Vec<search::Candidate>, Option<Vec<u64>>)> {
    let primary = if let Some((debt, gradient)) = &state.native_track {
        search::track_candidates(
            source,
            &state.base_gradient,
            debt,
            gradient,
            &state.seen,
            &mut state.budget,
        )?
    } else {
        search::track_candidates(
            source,
            &state.base_gradient,
            &state.debt,
            &state.gradient,
            &state.seen,
            &mut state.budget,
        )?
    };
    let preferred = state
        .native_track
        .as_ref()
        .and_then(|_| primary.first().map(|c| c.key()));
    let distinct_ay = state.ay_track_valid
        && state
            .native_track
            .as_ref()
            .is_some_and(|(debt, gradient)| debt != &state.debt || gradient != &state.gradient);
    // Precharge the bounded merge before optional work, so an exhausted
    // second track cannot make us discard an already generated native pool.
    let merge_work = source
        .spec
        .inputs
        .checked_add(MAX_SUPPORT)?
        .checked_mul(2)?
        .checked_add(1)?
        .checked_mul(MAX_CANDIDATES.checked_mul(2)?)?;
    if state.budget.charge(merge_work).is_none() {
        return Some((primary, preferred));
    }
    // Both searches spend the SAME work ledger and original deadline.
    let secondary = if distinct_ay {
        search::track_candidates(
            source,
            &state.base_gradient,
            &state.debt,
            &state.gradient,
            &state.seen,
            &mut state.budget,
        )
        .unwrap_or_default()
    } else {
        vec![]
    };
    let mut result = Vec::new();
    let mut keys = Vec::new();
    for at in 0..primary.len().max(secondary.len()) {
        for candidate in [primary.get(at), secondary.get(at)].into_iter().flatten() {
            let key = candidate.key();
            if !keys.contains(&key) {
                keys.push(key);
                result.push(candidate.clone());
                if result.len() == MAX_CANDIDATES {
                    return Some((result, preferred));
                }
            }
        }
    }
    if result.is_empty() {
        let original: Vec<_> = source.q.iter().map(|q| 0.5 * q.lo + 0.5 * q.hi).collect();
        result = search::track_candidates(
            source,
            &state.base_gradient,
            &original,
            &state.gradient,
            &state.seen,
            &mut state.budget,
        )?;
    }
    Some((result, preferred))
}

fn certify_batch<'a>(
    source: &'a Source,
    state: &mut Portfolio<'a>,
    round: &mut RoundReport,
) -> Option<Vec<CertifiedDirection<'a>>> {
    let limit = MAX_CUTS
        .saturating_sub(round.certified)
        .min(state.slots.checked_sub(state.accepted.len())?);
    if limit == 0 {
        return Some(vec![]);
    }
    if let Some(cut) = admission::first_signed(source, state, round) {
        round.certified += 1;
        return Some(vec![cut]);
    }
    let first_legacy_attempt = round.attempted;
    let (mut proposed, preferred) = candidates(source, state)?;
    proposed.truncate(MAX_CANDIDATES.saturating_sub(round.candidates));
    round.candidates += proposed.len();
    let mut used = vec![false; source.spec.heads];
    for cut in &state.accepted {
        for &(head, _) in &cut.direction.heads {
            used[head] = true;
        }
    }
    let mut cuts = Vec::new();
    while !proposed.is_empty() && cuts.len() < limit && state.report.attempted < MAX_ATTEMPTS {
        if state.budget.charge(proposed.len()).is_none() {
            break;
        }
        // Give the leading completed-native track one certification attempt
        // before shared novelty ranking; a stronger AY bound need not supply
        // a better direction for discovering the next independently proved row.
        let first_native = (round.attempted == first_legacy_attempt)
            .then(|| {
                preferred
                    .as_ref()
                    .and_then(|key| proposed.iter().position(|c| &c.key() == key))
            })
            .flatten();
        let at = first_native.or_else(|| {
            proposed
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| {
                    let fresh = |c: &search::Candidate| {
                        c.direction.heads.iter().filter(|&&(j, _)| !used[j]).count()
                    };
                    (fresh(a) * b.direction.heads.len())
                        .cmp(&(fresh(b) * a.direction.heads.len()))
                        .then_with(|| ib.cmp(ia))
                })
                .map(|(at, _)| at)
        })?;
        let candidate = proposed.remove(at);
        state.seen.push(candidate.key());
        state.report.attempted += 1;
        round.attempted += 1;
        if let Some(cut) = certify_affine(
            source,
            candidate.direction,
            candidate.eta,
            &mut state.budget,
        ) {
            for &(j, _) in &cut.direction.heads {
                used[j] = true;
            }
            cuts.push(cut);
        }
    }
    round.certified += cuts.len();
    Some(cuts)
}

fn replay_now(
    source: &Source,
    model: &triangle::Model<'_>,
    hint: &HeadFunctionalLpHint,
    sign: f64,
    is_native: bool,
    state: &mut Portfolio<'_>,
    deadline: Instant,
    authority: Instant,
) -> bool {
    if state.allocated_report.is_some() {
        let Some(terms) = model
            .request()
            .rows
            .iter()
            .try_fold(model.request().bounds.len(), |n, row| {
                n.checked_add(row.coefficients.len())
            })
        else {
            return false;
        };
        if state
            .budget
            .publication_charge(terms.saturating_mul(2), state.publication)
            .is_none()
        {
            return false;
        }
    }
    let Some(bound) = model.replay(source, hint, sign, deadline) else {
        return false;
    };
    state.report.lower = commit_triangle_best(state.report.lower, bound, deadline, authority);
    // A successful scalar replay is sufficient for search, even if it did not
    // improve. No imported dual receives authority to create a geometry row.
    if is_native || bound > state.search_bound {
        if let Some((debt, gradient)) = geometry_debt(
            model.request(),
            model.scalar_count(),
            model.scalar_count(),
            hint,
            sign,
            &mut state.budget,
        ) {
            if is_native {
                state.native_track = Some((debt, gradient));
            } else {
                state.debt = debt;
                state.gradient = gradient;
                state.search_bound = bound;
                state.ay_track_valid = true;
            }
        }
    }
    true
}

fn solve_round<F>(
    source: &Source,
    model: &triangle::Model<'_>,
    state: &mut Portfolio<'_>,
    round: &mut RoundReport,
    deadline: Instant,
    authority: Instant,
    provider: &F,
) where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    if state.allocated_report.is_some() {
        let Some(terms) = model
            .request()
            .rows
            .iter()
            .try_fold(model.request().bounds.len(), |n, row| {
                n.checked_add(row.coefficients.len())
            })
        else {
            return;
        };
        // Extra full-model ranking/export scan credit for the multirow path;
        // existing per-candidate charges and each actual replay remain charged.
        if state
            .budget
            .publication_charge(terms.saturating_mul(4), state.publication)
            .is_none()
        {
            return;
        }
    }
    if let Some(hint) = native::portfolio_hint(
        model,
        &state.accepted,
        &state.base_gradient,
        &mut state.budget,
    ) {
        if replay_now(source, model, &hint, 1.0, true, state, deadline, authority) {
            round.native_replayed += 1;
        }
    }
    // Native completed floors are committed BEFORE a possibly late callback.
    if state.report.lower > 0.0 || !live(state.budget.deadline) {
        return;
    }
    let hint = (|| {
        let provider = std::panic::catch_unwind(std::panic::AssertUnwindSafe(provider))
            .ok()
            .flatten()?;
        if !std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            provider.supports_functional_advice()
        }))
        .ok()?
            || !live(state.budget.deadline)
        {
            return None;
        }
        round.calls += 1;
        let hint = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            provider.propose_functional(model.request(), state.budget.deadline)
        }))
        .ok()
        .flatten()?;
        (live(state.budget.deadline) && hint.validate(model.request())).then_some(hint)
    })();
    if let Some(hint) = hint {
        for sign in [1.0, -1.0] {
            if replay_now(
                source, model, &hint, sign, false, state, deadline, authority,
            ) {
                round.ay_replayed += 1;
            }
        }
    }
}

fn record_round(
    state: &mut Portfolio<'_>,
    mut round: RoundReport,
    started: Instant,
    masks: usize,
    cells: usize,
) {
    round.after = state.report.lower;
    round.cuts = state.accepted.len() - state.report.cuts;
    state.report.cuts = state.accepted.len();
    round.masks = state.budget.masks - masks;
    round.work_cells = state.budget.cells - cells;
    round.elapsed_seconds = started.elapsed().as_secs_f64();
    eprintln!("[input-head-directional-round] round={} candidates={} attempted={} certified={} cuts={} algebraic_masks={} work_cells={} calls={} native_replayed={} ay_replayed={} before_bits={:016x} after_bits={:016x} elapsed_s={:.6}",
        round.round, round.candidates, round.attempted, round.certified, round.cuts,
        round.masks, round.work_cells, round.calls, round.native_replayed, round.ay_replayed,
        round.before.to_bits(), round.after.to_bits(), round.elapsed_seconds);
    state.report.candidates += round.candidates;
    state.report.certified += round.certified;
    state.report.calls += round.calls;
    state.report.native_replayed += round.native_replayed;
    state.report.ay_replayed += round.ay_replayed;
    state.report.rounds.push(round);
}

pub(in super::super) fn refine_with_prefix<'a, F>(
    source: &'a Source,
    before: f64,
    prefix: Option<triangle::PhasePrefix<'a>>,
    stage_deadline: Instant,
    authority: Instant,
    provider: &F,
) -> (Refinement, Option<triangle::PhasePrefix<'a>>)
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    let started = Instant::now();
    let deadline = stage_deadline.min(authority);
    let Some(call_deadline) = deadline.checked_sub(REPLAY_RESERVE) else {
        return (
            Refinement {
                lower: before,
                ..Refinement::default()
            },
            prefix,
        );
    };
    let mut state = Portfolio {
        retained: prefix,
        accepted: vec![],
        seen: vec![],
        debt: source.q.iter().map(|q| 0.5 * q.lo + 0.5 * q.hi).collect(),
        gradient: vec![],
        base_gradient: vec![],
        native_track: None,
        ay_track_valid: false,
        budget: Budget {
            deadline: call_deadline,
            cells: 0,
            masks: 0,
        },
        report: Refinement {
            lower: before,
            ..Refinement::default()
        },
        slots: 0,
        search_bound: f64::NEG_INFINITY,
        signed_attempted: false,
        signed_report: None,
        publication: PublicationAuthority { deadline },
        allocated_checked: false,
        allocated: vec![],
        allocated_next: 0,
        allocated_report: None,
    };
    let _ = (|| {
        if !good(before) || !live(call_deadline) {
            return None;
        }
        state.gradient = search::initial_gradient(source, &mut state.budget)?;
        state.base_gradient = state.gradient.clone();
        state.slots = triangle::Model::prepare_root_prefix(
            source,
            &mut state.retained,
            MAX_CUTS * MAX_ROUNDS,
            call_deadline,
        )?;
        for at in 0..MAX_ROUNDS {
            #[cfg(test)]
            if EXPIRE_AFTER_ROUND.with(|fault| fault.get() == Some(at)) {
                state.budget.deadline = Instant::now().checked_sub(Duration::from_millis(1))?;
            }
            if !live(state.budget.deadline)
                || state.report.lower > 0.0
                || state.accepted.len() >= state.slots
                || state.report.attempted >= MAX_ATTEMPTS
            {
                break;
            }
            state.search_bound = f64::NEG_INFINITY;
            let round_started = Instant::now();
            let (masks, cells) = (state.budget.masks, state.budget.cells);
            let mut round = RoundReport {
                round: at + 1,
                before: state.report.lower,
                ..RoundReport::default()
            };
            let _ = (|| {
                let model =
                    if let Some(model) = allocated_round::run(source, &mut state, &mut round) {
                        model
                    } else {
                        let cuts = certify_batch(source, &mut state, &mut round)?;
                        admission::admit(source, &mut state, cuts)?
                    };
                solve_round(
                    source, &model, &mut state, &mut round, deadline, authority, provider,
                );
                // The rows were fully admitted before the callback. Preserve
                // them even if that callback expired; no new proof is minted.
                state.retained = Some(model.into_phase_prefix());
                Some(())
            })();
            record_round(&mut state, round, round_started, masks, cells);
        }
        Some(())
    })();
    state.report.cuts = state.accepted.len();
    state.report.masks = state.budget.masks;
    state.report.work_cells = state.budget.cells;
    state.report.elapsed_seconds = started.elapsed().as_secs_f64();
    if let Some(receipt) = &state.signed_report {
        receipt.emit();
    }
    if let Some(receipt) = &state.allocated_report {
        receipt.emit();
    }
    (state.report, state.retained)
}
