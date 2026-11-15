// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One prepaid simultaneous direction trial. A partial bundle is advice only.
use super::*;

pub(super) struct Active {
    rows: [[usize; 2]; 2],
    count: [usize; 2],
    lower: [f64; 2],
}
impl Active {
    pub(super) fn new() -> Self {
        Self {
            rows: [[usize::MAX; 2]; 2],
            count: [0; 2],
            lower: [f64::INFINITY; 2],
        }
    }
    pub(super) fn record(&mut self, part: usize, row: usize, value: f64) {
        if value < self.lower[part] {
            self.lower[part] = value;
            self.rows[part][0] = row;
            self.count[part] = 1;
        } else if value == self.lower[part] && self.count[part] < 2 {
            self.rows[part][self.count[part]] = row;
            self.count[part] += 1;
        }
    }
}

fn claim(
    cuts: &HullRootMipCuts<'_>,
    choice: Choice,
    b: &Budget,
    ticket: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<usize> {
    ticket.alive(cuts, b, poll)?;
    if !ticket.preserved_mode || !ticket.transported_payload || !choice.transported {
        return None;
    }
    let mut state = ticket.state.lock().ok()?;
    if !state.paid_choice
        || !state.build
        || !state.transported_started
        || state.choice != Some(choice.bits())
        || state.oblique_choice != Some(choice.bits())
        || state.oblique_fee == 0
        || state.oblique_started
    {
        return None;
    }
    state.oblique_started = true;
    Some(state.oblique_fee)
}

// Four exact segment minimizations approximate the minimum-norm point of
// at most four sampled gradients. Missing active masks/corner ties remain
// missing: neither this vector nor its norm is a proof of global ascent.
fn min_norm(
    gradients: &[f64],
    count: usize,
    d: usize,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<f64>> {
    if !(1..=4).contains(&count) || d == 0 || gradients.len() != count.checked_mul(d)? || !poll() {
        return None;
    }
    let mut direction = gradients[..d].to_vec();
    for _ in 0..4 {
        if !poll() {
            return None;
        }
        let mut norm = 0.;
        for &x in &direction {
            norm = finite(norm + finite(x * x)?)?;
        }
        let mut best = f64::INFINITY;
        let mut chosen = 0;
        for sample in 0..count {
            let mut dot = 0.;
            for i in 0..d {
                dot = finite(dot + finite(direction[i] * gradients[sample * d + i])?)?;
            }
            if dot < best {
                best = dot;
                chosen = sample;
            }
        }
        let mut denominator = 0.;
        for i in 0..d {
            let difference = finite(direction[i] - gradients[chosen * d + i])?;
            denominator = finite(denominator + finite(difference * difference)?)?;
        }
        if denominator == 0. {
            break;
        }
        let amount = finite(finite(norm - best)? / denominator)?.clamp(0., 1.);
        if amount == 0. {
            break;
        }
        for i in 0..d {
            let difference = finite(gradients[chosen * d + i] - direction[i])?;
            direction[i] = finite(direction[i] + finite(amount * difference)?)?;
        }
    }
    let mut norm = 0.;
    for &x in &direction {
        norm = finite(norm + finite(x * x)?)?;
    }
    (norm > 0.).then_some(direction)
}

fn step(cache: &Cache, active: &Active, direction: &[f64], d: usize) -> Option<f64> {
    let width = d.checked_add(1)?;
    let mut first = f64::INFINITY;
    for part in 0..2 {
        for &mask in &active.rows[part][..active.count[part]] {
            let row = mask.checked_mul(width)?;
            for i in 0..d {
                let v = direction[i];
                if v == 0. {
                    continue;
                }
                let q = cache.delta[row + i + 1];
                let crossing = finite(if part == 0 { q / v } else { -q / v })?;
                if crossing > 0. {
                    first = first.min(crossing);
                }
            }
        }
    }
    let amount = finite(0.5 * first)?;
    (amount > 0.).then_some(amount)
}

pub(super) fn improve(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    choice: Choice,
    cache: &Cache,
    active: &Active,
    face: &mut Proposal,
    b: &Budget,
    ticket: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<bool> {
    let fee = claim(cuts, choice, b, ticket, poll)?;
    let p = cuts.problem;
    let d = p.input_dim;
    let width = d.checked_add(1)?;
    let masks = face.endpoint.len();
    let ka = plan.components.get(choice.a)?.negative.len();
    let kb = plan.components.get(choice.b)?.negative.len();
    if d == 0
        || plan.components.len() != ticket.old_components
        || plan.leftover.len() > ticket.dimensions.1
        || !(1..=8).contains(&ka)
        || !(1..=8).contains(&kb)
        || cache.split != 1usize << ka
        || masks != (1usize << ka) + (1usize << kb)
        || quote::oblique_extra(plan, d, choice.a, choice.b)? != fee
        || face.targets.len() != d.checked_mul(2)?
        || cache.delta.len() != masks.checked_mul(width)?
        || active.count.iter().any(|n| !(1..=2).contains(n))
        || active.rows[0][..active.count[0]]
            .iter()
            .any(|m| *m >= cache.split)
        || active.rows[1][..active.count[1]]
            .iter()
            .any(|m| *m < cache.split || *m >= masks)
    {
        return None;
    }
    let count = active.count[0] * active.count[1];
    // At most four D-vectors plus one D-vector from min_norm: <=5D.
    let mut gradients = vec![0.; count.checked_mul(d)?];
    let mut sample = 0;
    for &ma in &active.rows[0][..active.count[0]] {
        for &mb in &active.rows[1][..active.count[1]] {
            ticket.alive(cuts, b, poll)?;
            for i in 0..d {
                let qa = cache.delta[ma * width + i + 1];
                let qb = cache.delta[mb * width + i + 1];
                // Explicit lower-endpoint tie convention.
                let xa = if qa >= 0. { p.lower[i] } else { p.upper[i] };
                let xb = if qb >= 0. { p.lower[i] } else { p.upper[i] };
                gradients[sample * d + i] = finite(xb - xa)?;
            }
            sample += 1;
        }
    }
    let mut permit = || ticket.alive(cuts, b, poll).is_some();
    let direction = min_norm(&gradients, count, d, &mut permit)?;
    // Mandatory lifetime barrier before allocating a second target pair.
    drop(gradients);
    ticket.alive(cuts, b, poll)?;
    let amount = step(cache, active, &direction, d)?;
    // Retained face targets are already in the transported payload. Scratch
    // is now D direction +2D trial targets +2W global/zero, <=6W.
    let mut targets = vec![0.; d.checked_mul(2)?];
    for i in 0..d {
        let change = finite(amount * direction[i])?;
        targets[i] = finite(face.targets[i] + change)?;
        targets[d + i] = finite(face.targets[d + i] - change)?;
    }
    drop(direction);
    let mut global = vec![0.; width];
    for (index, c) in plan.components.iter().enumerate() {
        ticket.alive(cuts, b, poll)?;
        for i in 0..d {
            let g = if index == choice.a {
                targets[i]
            } else if index == choice.b {
                targets[d + i]
            } else {
                *c.g.get(i)?
            };
            global[i + 1] = finite(global[i + 1] + g)?;
        }
    }
    for &j in &plan.leftover {
        ticket.alive(cuts, b, poll)?;
        if j >= p.hidden_dim {
            return None;
        }
        add_head(
            p,
            &mut global,
            j,
            finite(p.c[j] * sanitize_alpha(cuts.multipliers.alpha[j]))?,
        )?;
    }
    let mut zero = vec![0.; width];
    let mut score = nominal(p, &global, &zero, 0.)?;
    // The global score is complete. Reuse these two paid scratch arrays
    // for actual stored changes, avoiding two subtractions per mask/axis.
    for i in 0..d {
        global[i + 1] = finite(targets[i] - face.targets[i])?;
        zero[i + 1] = finite(targets[d + i] - face.targets[d + i])?;
    }
    for (part, begin, end) in [(0, 0, cache.split), (1, cache.split, masks)] {
        let mut lower = f64::INFINITY;
        let change = if part == 0 { &global } else { &zero };
        for mask in begin..end {
            ticket.alive(cuts, b, poll)?;
            let row = mask * width;
            let mut value = cache.delta[row];
            for i in 0..d {
                // Use each owner's actual stored rounded change, not the
                // symbolic opposite change used to propose the trial.
                let q = finite(cache.delta[row + i + 1] - change[i + 1])?;
                value = finite(value + (q * p.lower[i]).min(q * p.upper[i]))?;
            }
            lower = lower.min(value);
        }
        score = finite(score + lower)?;
    }
    ticket.alive(cuts, b, poll)?;
    if score > face.score {
        // One endpoint table, no residual clone, and no extra Source check.
        face.targets = targets;
        face.score = score;
        // The old axis telemetry must not misdescribe a simultaneous move.
        face.axis = None;
        face.stored_changes = [0.; 2];
        Some(true)
    } else {
        Some(false)
    }
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_oblique_tests.rs"]
mod tests;
