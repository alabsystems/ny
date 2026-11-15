// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Uniform ownership transport and a bounded exact-coordinate face oracle.
//! All caches/faces are nominal advice. Only the whole Source check certifies.
use super::*;
#[path = "single_hidden_hull_mip_signed_root_coupling_oblique.rs"]
mod oblique;

pub(super) struct Proposal {
    pub(super) endpoint: Vec<f64>,
    pub(super) targets: Vec<f64>,
    pub(super) score: f64,
    pub(super) axis: Option<usize>,
    pub(super) stored_changes: [f64; 2],
}

pub(super) fn eligible(cuts: &HullRootMipCuts<'_>, plan: &Plan, c: Choice) -> bool {
    let a = &plan.components[c.a];
    let b = &plan.components[c.b];
    let matched = |part: &Component, slot: usize| {
        part.alpha[slot].to_bits()
            == sanitize_alpha(cuts.multipliers.alpha[part.positive[slot]]).to_bits()
    };
    c.fast[0] && matched(a, c.sa) && c.sb.is_none_or(|slot| c.fast[1] && matched(b, slot))
}

fn directions(cuts: &HullRootMipCuts<'_>, plan: &Plan, c: Choice) -> Option<Vec<f64>> {
    let p = cuts.problem;
    let a = &plan.components[c.a];
    let b = &plan.components[c.b];
    let j = a.positive[c.sa];
    let aj = sanitize_alpha(cuts.multipliers.alpha[j]);
    let cj = finite(p.c[j] * aj)?;
    let incoming = c.sb.map(|slot| {
        let k = b.positive[slot];
        (k, sanitize_alpha(cuts.multipliers.alpha[k]))
    });
    let mut targets = vec![0.; p.input_dim.checked_mul(2)?];
    for i in 0..p.input_dim {
        let transfer = match incoming {
            None => finite(-finite(0.5 * cj)? * p.w_row(j)[i])?,
            Some((k, ak)) => {
                finite(finite(finite(p.c[k] * ak)? * p.w_row(k)[i])? - finite(cj * p.w_row(j)[i])?)?
            }
        };
        targets[i] = finite(a.g[i] + transfer)?;
        targets[p.input_dim + i] = finite(b.g[i] - transfer)?;
    }
    Some(targets)
}

#[derive(Clone, Copy)]
struct Axis {
    index: usize,
    score: f64,
    // [A lower face, A upper face, B lower face, B upper face]
    faces: [f64; 4],
}
impl Axis {
    const EMPTY: Self = Self {
        index: usize::MAX,
        score: f64::NEG_INFINITY,
        faces: [f64::INFINITY; 4],
    };
}

fn axes(p: &HullMarginProblem, targets: &[f64]) -> Option<([Axis; 4], usize)> {
    let mut result = [Axis::EMPTY; 4];
    let mut count = 0usize;
    for i in 0..p.input_dim {
        let width = finite(p.upper[i] - p.lower[i])?;
        if width <= 0. {
            continue;
        }
        let score = finite(width * finite(targets[i] - targets[p.input_dim + i])?.abs())?;
        let candidate = Axis {
            index: i,
            score,
            ..Axis::EMPTY
        };
        let position = (0..count)
            .find(|&slot| score > result[slot].score)
            .unwrap_or(count);
        if position >= 4 {
            continue;
        }
        for slot in (position + 1..=count.min(3)).rev() {
            result[slot] = result[slot - 1];
        }
        result[position] = candidate;
        count = (count + 1).min(4);
    }
    Some((result, count))
}

fn best_exchange(lower: f64, upper: f64, faces: [f64; 4]) -> Option<(f64, f64)> {
    let width = finite(upper - lower)?;
    if width <= 0. {
        return None;
    }
    let x = finite(finite(faces[1] - faces[0])? / width)?;
    let y = finite(finite(faces[2] - faces[3])? / width)?;
    // Closest-to-zero point in the exact maximizing interval.
    let lo = x.min(y);
    let hi = x.max(y);
    let s = if lo > 0. {
        lo
    } else if hi < 0. {
        hi
    } else {
        0.
    };
    let a = finite(faces[0] - finite(lower * s)?)?.min(finite(faces[1] - finite(upper * s)?)?);
    let b = finite(faces[2] + finite(lower * s)?)?.min(finite(faces[3] + finite(upper * s)?)?);
    Some((s, finite(a + b)?))
}

pub(super) fn propose(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    choice: Choice,
    cache: &mut Cache,
    b: &Budget,
    ticket: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Proposal> {
    ticket.alive(cuts, b, poll)?;
    if !choice.transported || !ticket.transported_payload || !eligible(cuts, plan, choice) {
        return None;
    }
    let oblique_funded = {
        let mut use_state = ticket.state.lock().ok()?;
        if use_state.transported_started
            || !use_state.paid_choice
            || !use_state.build
            || use_state.choice != Some(choice.bits())
        {
            return None;
        }
        use_state.transported_started = true;
        use_state.oblique_choice == Some(choice.bits())
    };
    let p = cuts.problem;
    let d = p.input_dim;
    let width = d.checked_add(1)?;
    let mut targets = directions(cuts, plan, choice)?;
    let (mut axes, count) = axes(p, &targets)?;
    let mut endpoint = vec![0.; cache.endpoint.len()];
    let (ea, eb) = edits(plan, choice);
    let mut v = vec![0.; width];
    let mut residual = vec![0.; width];
    let mut pair_lower = [f64::INFINITY; 2];
    for (part, index, edit, offset) in [(0, choice.a, ea, 0), (1, choice.b, eb, cache.split)] {
        let c = &plan.components[index];
        v.fill(0.);
        let coefficient = finite(p.c[edit.head] * (f64::from(edit.units) * 0.5))?;
        add_head(p, &mut v, edit.head, coefficient)?;
        let mut dominant = 0;
        let mut importance = 0.;
        for i in 0..d {
            let score = finite(v[i + 1].abs() * finite(p.upper[i] - p.lower[i])?)?;
            if score > importance {
                dominant = i;
                importance = score;
            }
        }
        for mask in 0..1usize << c.negative.len() {
            ticket.alive(cuts, b, poll)?;
            let row = (offset + mask) * width;
            let old = seed(cuts, c, edit, mask);
            residual[0] = finite(cache.start[row] - old * v[0])?;
            for i in 0..d {
                residual[i + 1] = finite(
                    cache.start[row + i + 1] - old * v[i + 1] + c.g[i] - targets[part * d + i],
                )?;
            }
            let crossing = if importance > 0. {
                finite((-residual[dominant + 1] / v[dominant + 1]).clamp(0., 1.))?
            } else {
                old
            };
            let mut selected = old;
            let mut best = nominal(p, &residual, &v, old)?;
            for alpha in [0., 1., crossing] {
                let score = nominal(p, &residual, &v, alpha)?;
                if score > best {
                    selected = alpha;
                    best = score;
                }
            }
            endpoint[offset + mask] = selected;
            pair_lower[part] = pair_lower[part].min(best);
            // The canonical winner no longer needs its delta cache.
            // Store complete new residuals for a separate direct reprice.
            for i in 0..width {
                cache.delta[row + i] = finite(residual[i] + selected * v[i])?;
            }
            for axis in &mut axes[..count] {
                let i = axis.index;
                let q = cache.delta[row + i + 1];
                let left = finite(q * p.lower[i])?;
                let right = finite(q * p.upper[i])?;
                let other = finite(best - left.min(right))?;
                axis.faces[2 * part] = axis.faces[2 * part].min(finite(other + left)?);
                axis.faces[2 * part + 1] = axis.faces[2 * part + 1].min(finite(other + right)?);
            }
            note(b, |r| {
                r.line_mask_attempts += 1;
                r.line_masks_completed += 1;
                r.line_masks_improved += usize::from(selected != old);
            });
        }
    }
    drop(v);
    drop(residual);
    let mut best = finite(pair_lower[0] + pair_lower[1])?;
    let mut chosen = None;
    for axis in &axes[..count] {
        ticket.alive(cuts, b, poll)?;
        let Some((s, score)) = best_exchange(p.lower[axis.index], p.upper[axis.index], axis.faces)
        else {
            continue;
        };
        // Reject a nominal exchange that cannot be represented as stored g.
        let ga = finite(targets[axis.index] + s);
        let gb = finite(targets[d + axis.index] - s);
        if let (Some(ga), Some(gb)) = (ga, gb) {
            if score > best {
                best = score;
                chosen = Some((axis.index, ga, gb));
            }
        }
    }
    let mut actual_change = [0.; 2];
    let mut changed_axis = None;
    if let Some((i, ga, gb)) = chosen {
        actual_change[0] = finite(ga - targets[i])?;
        actual_change[1] = finite(gb - targets[d + i])?;
        targets[i] = ga;
        targets[d + i] = gb;
        changed_axis = Some(i);
    }
    // Evaluate stored directions, including their actual floating sum.
    // No symbolic "the sum is unchanged" shortcut supplies the global term.
    let mut global = vec![0.; width];
    for (index, c) in plan.components.iter().enumerate() {
        ticket.alive(cuts, b, poll)?;
        for i in 0..d {
            let g = if index == choice.a {
                targets[i]
            } else if index == choice.b {
                targets[d + i]
            } else {
                c.g[i]
            };
            global[i + 1] = finite(global[i + 1] + g)?;
        }
    }
    for &j in &plan.leftover {
        ticket.alive(cuts, b, poll)?;
        add_head(
            p,
            &mut global,
            j,
            finite(p.c[j] * sanitize_alpha(cuts.multipliers.alpha[j]))?,
        )?;
    }
    let zero = vec![0.; width];
    let mut score = nominal(p, &global, &zero, 0.)?;
    let mut active = oblique::Active::new();
    for (part, begin, end) in [(0, 0, cache.split), (1, cache.split, endpoint.len())] {
        let mut lower = f64::INFINITY;
        for mask in begin..end {
            ticket.alive(cuts, b, poll)?;
            let row = mask * width;
            if let Some(i) = changed_axis {
                cache.delta[row + i + 1] = finite(cache.delta[row + i + 1] - actual_change[part])?;
            }
            let value = nominal(p, &cache.delta[row..row + width], &zero, 0.)?;
            if oblique_funded {
                active.record(part, mask, value);
            }
            lower = lower.min(value);
        }
        score = finite(score + lower)?;
    }
    // Reuse the paid six-width scratch phase; these arrays must not coexist
    // with the four-gradient bundle and its direction.
    drop(global);
    drop(zero);
    let mut proposal = Proposal {
        endpoint,
        targets,
        score,
        axis: changed_axis,
        stored_changes: actual_change,
    };
    if oblique_funded {
        let before = proposal.score;
        let changed = oblique::improve(
            cuts,
            plan,
            choice,
            cache,
            &active,
            &mut proposal,
            b,
            ticket,
            poll,
        );
        // Numerical refusal keeps the face proposal. A cancelled or expired
        // original ticket cannot proceed to fresh checking.
        ticket.alive(cuts, b, poll)?;
        eprintln!(
            "[mip-root-oblique] funded=1 improved={:?} face={:?} proposed={:?}",
            changed, before, proposal.score,
        );
    }
    Some(proposal)
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_transported_tests.rs"]
mod tests;
