// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Nominal joint-line/coordinate schedule inside the immutable paid allowance.
use super::*;
#[path = "single_hidden_hull_mip_signed_root_coupling_line.rs"]
mod line;

/// Caller has consumed the complete whole-plan/search quote. This helper never
/// calls global Budget.charge/proof and never creates a clock or proof permit.
pub(super) fn optimize(
    cuts: &HullRootMipCuts<'_>,
    part: &mut Component,
    active: usize,
    role: SearchRole,
    b: &Budget,
    t: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<usize> {
    t.claim_search(cuts, b, part, active.min(part.positive.len()), role)?;
    let p = cuts.problem;
    let active = active.min(part.positive.len());
    if part.positive.is_empty() {
        return Some(0);
    }
    let schedule = if matches!(role, SearchRole::Selected) && part.positive.len() >= 2 {
        line::Schedule::new(
            p.input_dim,
            part.negative.len(),
            part.positive.len(),
            active,
        )?
    } else {
        line::Schedule::legacy(active)
    };
    let mut joint = schedule.enabled.then(|| line::Run::new(schedule, p, t.end));
    let original = part.clone();
    let masks = 1usize.checked_shl(part.negative.len() as u32)?;
    if part.alpha.len() != masks.checked_mul(part.positive.len())? {
        return None;
    }
    for mask in 0..masks {
        t.alive(cuts, b, poll)?;
        let alpha =
            &mut part.alpha[mask * original.positive.len()..(mask + 1) * original.positive.len()];
        let (mut q, mut constant) = search::mask_cache(p, &original, &part.g, mask, alpha)?;
        if schedule.enabled {
            note(b, |r| r.line_mask_attempts += 1);
            let outcome = joint.as_mut()?.apply(
                p,
                &original,
                &part.g,
                mask,
                alpha,
                &mut q,
                &mut constant,
                t.end,
                &mut || t.alive(cuts, b, poll).is_some(),
            )?;
            let mut state = b.state.lock().ok()?;
            state.coordinate_proposals =
                state.coordinate_proposals.checked_add(outcome.proposals)?;
            state.root_coupling.line_masks_completed += usize::from(outcome.completed);
            state.root_coupling.line_masks_improved += usize::from(outcome.improved);
        }
        for _ in 0..schedule.sweeps {
            for (slot, &j) in original.positive.iter().enumerate().take(schedule.visits) {
                t.alive(cuts, b, poll)?;
                let previous = alpha[slot];
                let weight = original.nominal_coefficient(p, slot)?;
                let cb = finite(weight * p.b1[j])?;
                let base = finite(constant - previous * cb)?;
                let direction: Vec<_> = p
                    .w_row(j)
                    .iter()
                    .map(|w| finite(weight * w))
                    .collect::<Option<_>>()?;
                let residual: Vec<_> = q
                    .iter()
                    .zip(&direction)
                    .map(|(v, w)| finite(v - previous * w))
                    .collect::<Option<_>>()?;
                let mut proposals = 0;
                let chosen =
                    if search::alpha::work(p.input_dim)? < search::legacy_work(p.input_dim)? {
                        search::alpha::choose(
                            p,
                            &residual,
                            &direction,
                            base,
                            cb,
                            previous,
                            &q,
                            constant,
                            &mut proposals,
                            t.end,
                            poll,
                        )
                    } else {
                        search::legacy_choose(
                            p,
                            &residual,
                            &direction,
                            base,
                            cb,
                            previous,
                            &q,
                            constant,
                            &mut proposals,
                        )
                    }?;
                alpha[slot] = chosen;
                constant = finite(base + chosen * cb)?;
                for i in 0..p.input_dim {
                    q[i] = finite(residual[i] + chosen * direction[i])?;
                }
                let mut c = b.state.lock().ok()?;
                c.coordinate_proposals = c.coordinate_proposals.checked_add(proposals)?;
                c.root_coupling.coordinate_visits += 1;
            }
        }
        t.alive(cuts, b, poll)?;
        b.state.lock().ok()?.completed_masks += 1;
    }
    Some(schedule.visits)
}
