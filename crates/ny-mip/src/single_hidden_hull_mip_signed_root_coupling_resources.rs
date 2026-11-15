// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Metadata-only complete-plan quotes. No pair-only pass substitutes for Source.
use super::*;

#[derive(Clone, Debug)]
pub(super) struct Choice {
    pub left: usize,
    pub right: usize,
    pub selected: usize,
    pub positive: usize,
    pub recipient: usize,
    pub active: usize,
    pub check: usize,
    pub search: usize,
    pub total: usize,
    pub score: f64,
}
impl Choice {
    pub(super) fn bits(&self) -> [u64; 10] {
        [
            self.left as u64,
            self.right as u64,
            self.selected as u64,
            self.positive as u64,
            self.recipient as u64,
            self.active as u64,
            self.check as u64,
            self.search as u64,
            self.total as u64,
            self.score.to_bits(),
        ]
    }
}
pub(super) fn words(plan: &Plan) -> Option<usize> {
    plan.components
        .iter()
        .try_fold(2usize.checked_add(plan.leftover.len())?, |n, c| {
            n.checked_add(5)?
                .checked_add(c.negative.len())?
                .checked_add(c.positive.len().checked_mul(2)?)?
                .checked_add(c.g.len())?
                .checked_add(c.alpha.len())
        })
}
pub(super) fn component_cost(d: usize, k: usize, p: usize, half: usize) -> Option<usize> {
    let m = 1usize.checked_shl(k as u32)?;
    m.checked_mul(d.checked_add(1)?)?
        .checked_mul(p.checked_add(k)?.checked_add(3)?)?
        .checked_mul(4)?
        .checked_add(m.checked_mul(half)?.checked_mul(4)?)
}
pub(super) fn search_cost(d: usize, k: usize, p: usize, active: usize) -> Option<usize> {
    if active > p {
        return None;
    }
    if p == 0 {
        return Some(0);
    }
    let m = 1usize.checked_shl(k as u32)?;
    m.checked_mul(d.checked_add(1)?)?
        .checked_mul(p.checked_add(k)?.checked_add(1)?)?
        .checked_mul(4)?
        .checked_add(
            m.checked_mul(2)?
                .checked_mul(active)?
                .checked_mul(search::coordinate_work(d)?)?,
        )
}
/// Permanent extra payload covers both plan alternatives and clones/seals,
/// metadata/features and the one-mask optimizer scratch; no full mask-D cache.
pub(super) fn admission(plan: &Plan, b: &Budget) -> Option<(usize, usize, usize)> {
    let (d, h, p) = b.dimensions;
    let c = plan.components.len();
    if !(2..=64).contains(&c) || c > h || p == 0 {
        return None;
    }
    let old = words(plan)?;
    let added = 4096usize.checked_add(14usize.checked_mul(d.checked_add(5)?)?)?;
    let new = old.checked_add(added)?;
    let sigma = signature_len_shape(d, h)?;
    let payload = payload::quote(plan, d, h, p, c, old, new)?;
    // Pay this complete bound before Source/head feature work or pair iteration.
    let meta = c
        .checked_add(h.checked_mul(2)?)?
        .checked_mul(d.checked_add(1)?)?
        .checked_add(c.checked_mul(c)?.checked_mul(d.checked_add(16)?)?)?
        .checked_add(h)?
        .checked_add(1)?
        .checked_mul(16)?
        .checked_add(old.checked_mul(32)?)?
        .checked_add(new.checked_add(sigma)?.checked_mul(8)?)?
        .checked_add(
            c.checked_add(14)?
                .checked_mul(d.checked_add(1)?)?
                .checked_add(h)?
                .checked_add(p)?
                .checked_add(17)?
                .checked_mul(32)?,
        )?
        .checked_add(32usize.checked_mul(16)?.checked_mul(d.checked_add(1)?)?)?
        .checked_add(16usize.checked_mul(16)?.checked_mul(6)?)?
        .checked_add(
            32usize
                .checked_mul(p.checked_mul(2)?.min(96))?
                .checked_mul(d.checked_add(1)?)?,
        )?
        .checked_add(
            16usize
                .checked_mul(p.checked_mul(2)?.min(96))?
                .checked_mul(16)?,
        )?;
    let meta = meta.checked_add(if p >= 2 { 256 } else { 0 })?;
    Some((meta, payload, new))
}
fn signature_len_shape(d: usize, h: usize) -> Option<usize> {
    d.checked_mul(h)?
        .checked_mul(2)?
        .checked_add(h.checked_mul(3)?)?
        .checked_add(d.checked_mul(2)?)?
        .checked_add(11)
}
fn half(c: &Component) -> usize {
    c.half_units()
}

/// Cached parent metadata. Combining a bounded parent set never rescans Source.
#[derive(Clone, Copy)]
struct Shape {
    k: usize,
    appearances: usize,
    halves: usize,
    cost: usize,
    masks: usize,
    count: usize,
}
impl Shape {
    fn plus(self, z: Self) -> Option<Self> {
        Some(Self {
            k: self.k.checked_add(z.k)?,
            appearances: self.appearances.checked_add(z.appearances)?,
            halves: self.halves.checked_add(z.halves)?,
            cost: self.cost.checked_add(z.cost)?,
            masks: self.masks.checked_add(z.masks)?,
            count: self.count.checked_add(z.count)?,
        })
    }
}
/// Complete whole-Plan quote from cached scalar aggregates only. A union of
/// whole-unit parents may return omitted positives to the global leftover.
/// Half-unit parents retain the original two-parent/one-recipient path.
fn quote(
    d: usize,
    h: usize,
    c: usize,
    old_masks: usize,
    old_cost: usize,
    free: usize,
    shape: Shape,
    left: usize,
    right: usize,
) -> Option<Choice> {
    let k = shape.k;
    let appearances = shape.appearances;
    if !(2..=6).contains(&shape.count)
        || !(2..=16).contains(&k)
        || appearances == 0
        || appearances > 96
        || shape.halves > 0 && (shape.count != 2 || appearances > 32)
    {
        return None;
    }
    let selected = k.min(6).min(if shape.halves > 0 && appearances > 16 {
        k - 1
    } else {
        k
    });
    let positive = appearances.min(16);
    let recipient = if shape.halves == 0 {
        0
    } else {
        appearances - positive
    };
    let tails = k - selected;
    let nc = c
        .checked_sub(shape.count)?
        .checked_add(1)?
        .checked_add(tails)?;
    let nm = old_masks
        .checked_sub(shape.masks)?
        .checked_add(1usize << selected)?
        .checked_add(tails.checked_mul(2)?)?;
    if nc > h || nm > MAX_MASKS {
        return None;
    }
    // The fixed whole-Source base is NEVER removed. It covers all newly
    // leftover original-positive affine terms and final sum roundoff.
    let check = old_cost
        .checked_sub(shape.cost)?
        .checked_add(component_cost(d, selected, positive, positive)?)?
        .checked_add(tails.checked_mul(component_cost(d, 1, 0, 0)?)?)?
        .checked_add(
            component_cost(d, 1, recipient, recipient)?.checked_sub(component_cost(d, 1, 0, 0)?)?,
        )?;
    let tail_search = search_cost(d, 1, recipient, recipient)?;
    let setup = search_cost(d, selected, positive, 0)?;
    let room = free
        .checked_sub(check)?
        .checked_sub(tail_search)?
        .checked_sub(setup)?;
    let per = (1usize << selected)
        .checked_mul(2)?
        .checked_mul(search::coordinate_work(d)?)?;
    let active = positive.min(6).min(room / per);
    if active < positive.min(3) {
        return None;
    }
    let search = setup
        .checked_add(per.checked_mul(active)?)?
        .checked_add(tail_search)?;
    let total = check.checked_add(search)?;
    Some(Choice {
        left,
        right,
        selected,
        positive,
        recipient,
        active,
        check,
        search,
        total,
        score: 0.,
    })
}
fn cancellation(a: &[f64], z: &[f64]) -> Option<f64> {
    a.iter().zip(z).try_fold(0., |sum, (a, z)| {
        finite(sum + (a.abs() + z.abs() - (a + z).abs()).max(0.))
    })
}
/// One paid Source feature pass, a seed tournament, then at most four paid
/// growth rounds. Only the final set is built/optimized/checked. Every losing
/// extension is scalar metadata plus O(D) cached-feature pricing.
pub(super) fn choose(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    ticket: &Ticket,
    old_cost: usize,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Choice> {
    ticket.claim_selection(cuts, b, plan, old_cost)?;
    let p = cuts.problem;
    let d = p.input_dim;
    let c = plan.components.len();
    let mut features = Vec::with_capacity(c);
    let mut shapes = Vec::with_capacity(c);
    for part in &plan.components {
        let mut v = vec![0.; d + 1];
        for (slot, &j) in part.positive.iter().enumerate() {
            ticket.alive(cuts, b, poll)?;
            let weight = part.nominal_coefficient(p, slot)?;
            for i in 0..d {
                v[i] = finite(v[i] + weight * p.w_row(j)[i].abs() * (p.upper[i] - p.lower[i]))?;
            }
            v[d] = finite(v[d] + weight * p.b1[j].abs())?;
        }
        for &j in &part.negative {
            ticket.alive(cuts, b, poll)?;
            for i in 0..d {
                v[i] = finite(v[i] + p.c[j] * p.w_row(j)[i].abs() * (p.upper[i] - p.lower[i]))?;
            }
            v[d] = finite(v[d] + p.c[j] * p.b1[j].abs())?;
        }
        shapes.push(Shape {
            k: part.negative.len(),
            appearances: part.positive.len(),
            halves: half(part),
            cost: component_cost(d, part.negative.len(), part.positive.len(), half(part))?,
            masks: 1usize << part.negative.len(),
            count: 1,
        });
        features.push(v);
    }
    let free = ticket.remaining(cuts, b)?;
    let old_masks = plan.masks()?;
    let mut best: Option<(Choice, [usize; 6], Shape)> = None;
    for left in 0..c {
        for right in left + 1..c {
            ticket.alive(cuts, b, poll)?;
            let shape = shapes[left].plus(shapes[right])?;
            let Some(mut choice) = quote(
                d,
                p.hidden_dim,
                c,
                old_masks,
                old_cost,
                free,
                shape,
                left,
                right,
            ) else {
                continue;
            };
            let score = finite(
                cancellation(&features[left], &features[right])? / (choice.total.max(1) as f64),
            )?;
            if score <= 0. {
                continue;
            }
            choice.score = score;
            if best.as_ref().is_none_or(|(old, _, _)| score > old.score) {
                best = Some((choice, [left, right, 0, 0, 0, 0], shape));
            }
        }
    }
    let (mut choice, mut parents, mut shape) = best?;
    let mut aggregate = vec![0.; d + 1];
    for &index in &parents[..2] {
        for i in 0..=d {
            aggregate[i] = finite(aggregate[i] + features[index][i])?;
        }
    }
    // No half-unit mass is converted into a leftover. Original two-parent
    // recipient behavior remains available, but only whole sets may grow.
    while shape.count < 6 && shape.k < 6 && shape.halves == 0 {
        let mut extension: Option<(Choice, usize, Shape)> = None;
        for index in 0..c {
            if parents[..shape.count].contains(&index) || shapes[index].halves > 0 {
                continue;
            }
            ticket.alive(cuts, b, poll)?;
            let next = shape.plus(shapes[index])?;
            let Some(mut candidate) = quote(
                d,
                p.hidden_dim,
                c,
                old_masks,
                old_cost,
                free,
                next,
                choice.left,
                choice.right,
            ) else {
                continue;
            };
            let gain = cancellation(&aggregate, &features[index])?;
            let score = finite(gain / (candidate.total.max(1) as f64))?;
            if score <= 0. {
                continue;
            }
            candidate.score = score;
            if extension
                .as_ref()
                .is_none_or(|(old, _, _)| score > old.score)
            {
                extension = Some((candidate, index, next));
            }
        }
        let Some((next, index, next_shape)) = extension else {
            break;
        };
        parents[shape.count] = index;
        // At most six indices: insertion preserves a canonical ascending seal.
        let mut at = shape.count;
        while at > 0 && parents[at] < parents[at - 1] {
            parents.swap(at, at - 1);
            at -= 1;
        }
        for i in 0..=d {
            aggregate[i] = finite(aggregate[i] + features[index][i])?;
        }
        shape = next_shape;
        choice = next;
        choice.left = parents[0];
        choice.right = parents[1];
    }
    ticket.bind_union_choice(cuts, b, &choice, &parents[..shape.count])?;
    Some(choice)
}
