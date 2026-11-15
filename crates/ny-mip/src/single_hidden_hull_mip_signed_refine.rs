// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Retained bounded direction/batch/current-and-zero refinement for either seed.
use super::*;
#[allow(clippy::too_many_arguments)]
pub(super) fn refine(
    cuts: &HullRootMipCuts<'_>,
    plan: &mut Plan,
    current: &mut f64,
    best: &mut Option<SignedMajorant>,
    floor: f64,
    cost: usize,
    b: &Budget,
    seal: &Arc<RetainedWords>,
    poll: &mut dyn FnMut() -> bool,
) {
    let p = cuts.problem;
    direction::refine(cuts, plan, current, best, floor, cost, b, seal, poll);
    let batched = batch::refine(cuts, plan, current, best, floor, cost, b, seal, poll);
    let (mut refined, pending_zero) = batched.map_or_else(
        || {
            (
                vec![false; plan.components.len()],
                vec![false; plan.components.len()],
            )
        },
        |done| (done.refined, done.pending_zero),
    );
    let mut order: Vec<_> = (0..plan.components.len())
        .filter(|&i| !plan.components[i].positive.is_empty())
        .collect();
    order.sort_by_key(|&i| std::cmp::Reverse(plan.components[i].positive.len()));
    'components: for index in order {
        // A positive current-G batch does not cancel its planned zero seeds.
        // Earlier checked plans survive any cap/deadline refusal of that advice.
        if *current > 0. && !pending_zero[index] {
            continue;
        }
        // Preserve current/consensus geometry before spending on optional zero.
        let retained = plan.components[index].g.clone();
        let mut seeds = if *current > 0. {
            Vec::new()
        } else {
            vec![retained.clone()]
        };
        if retained.iter().any(|v| *v != 0.) {
            seeds.push(vec![0.; p.input_dim]);
        }
        for g in seeds {
            if !live(b.search_end) || !poll() {
                b.stop("poll");
                break 'components;
            }
            let Some(component) = search::optimize(cuts, &plan.components[index], g, b, cost, poll)
            else {
                if !live(b.search_end) {
                    break 'components;
                } else {
                    continue;
                }
            };
            let mut trial = plan.clone();
            trial.components[index] = component;
            if b.proof(cost, false).is_none() {
                break 'components;
            }
            let Some(checked) = certificate::check(cuts, &trial, b, b.search_end, poll) else {
                b.stop("check");
                break 'components;
            };
            // Compare the WHOLE original-q floor, including sum-G roundoff.
            if checked.root_lower <= *current {
                continue;
            }
            *current = checked.root_lower;
            *plan = trial;
            if !refined[index] {
                refined[index] = true;
                if let Ok(mut c) = b.state.lock() {
                    c.refined_components += 1;
                }
            }
            if *current > floor {
                *best = Some(row(cuts, plan, checked, cost, b, seal));
            }
            // Finish this component's bounded seeds: a positive current-G
            // floor can still be weaker than its already-planned zero seed.
        }
        if *current > 0. {
            b.stop("positive");
        }
    }
}

pub(super) fn empty_report(reason: &'static str) -> SignedRootReport {
    SignedRootReport {
        direction: SignedDirectionReport::default(),
        root_coupling: SignedRootCouplingReport::default(),
        attempted: true,
        admitted: false,
        components: 0,
        refined_components: 0,
        assigned_positive_heads: 0,
        positive_appearances: 0,
        split_positive_heads: 0,
        mass: SignedMassReport::default(),
        component_rows: SignedComponentRowsReport::default(),
        negative_heads: 0,
        coordinate_proposals: 0,
        completed_masks: 0,
        checked_masks: 0,
        source_replays: 0,
        work_units: 0,
        admitted_bytes: 0,
        lower: None,
        stop_reason: reason,
    }
}
pub(super) fn report(
    b: &Budget,
    plan: Option<&Plan>,
    lower: Option<f64>,
    admitted: bool,
) -> SignedRootReport {
    let c = b.state.lock().expect("private signed ledger poisoned");
    SignedRootReport {
        attempted: true,
        admitted,
        components: plan.map_or(0, |p| p.components.len()),
        refined_components: c.refined_components,
        assigned_positive_heads: plan.map_or(0, |plan| b.dimensions.2 - plan.leftover.len()),
        positive_appearances: plan
            .map_or(0, |p| p.components.iter().map(|c| c.positive.len()).sum()),
        split_positive_heads: plan.map_or(0, |p| {
            let assigned = b.dimensions.2 - p.leftover.len();
            p.components
                .iter()
                .map(|c| c.positive.len())
                .sum::<usize>()
                .saturating_sub(assigned)
        }),
        mass: c.mass.clone(),
        component_rows: c.component_rows.clone(),
        negative_heads: plan.map_or(0, |p| p.components.iter().map(|c| c.negative.len()).sum()),
        coordinate_proposals: c.coordinate_proposals,
        completed_masks: c.completed_masks,
        checked_masks: c.checked_masks,
        source_replays: c.source_replays,
        work_units: c.work,
        admitted_bytes: b.admitted_bytes
            + c.shared_bytes
            + c.direction.cache_bytes
            + c.child.cache_bytes
            + c.component_rows.admitted_bytes,
        direction: c.direction.clone(),
        root_coupling: c.root_coupling.clone(),
        lower,
        stop_reason: c.reason.unwrap_or(if admitted {
            "completed"
        } else {
            "no_improvement"
        }),
    }
}
