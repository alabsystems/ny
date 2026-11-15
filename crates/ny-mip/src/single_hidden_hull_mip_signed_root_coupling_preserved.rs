// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Keep every original negative mask while moving positive mass and direction.
//! All floating point caches are advice; the unchanged whole Source checker
//! alone can replace the old checked owner.
use super::*;
#[path = "single_hidden_hull_mip_signed_root_coupling_preserved_resources.rs"]
mod quote;
#[path = "single_hidden_hull_mip_signed_root_coupling_transported.rs"]
mod transported;

pub(super) fn eligible(plan: &Plan) -> bool {
    (2..=64).contains(&plan.components.len())
        && plan.components.iter().any(|c| c.negative.len() > 6)
}
pub(super) fn admission_with_transport(
    plan: &Plan,
    b: &Budget,
) -> Option<(usize, usize, usize, bool)> {
    quote::admission_with_transport(plan, b)
}
fn capacity(plan: &Plan) -> Option<usize> {
    quote::capacity(plan)
}

#[derive(Clone, Copy)]
struct Choice {
    a: usize,
    b: usize,
    sa: usize,
    // None: split A's whole head between A/B. Some: exchange two whole heads.
    sb: Option<usize>,
    fast: [bool; 2],
    transported: bool,
    check: usize,
    search: usize,
    total: usize,
    score: f64,
}
impl Choice {
    fn bits(self) -> [u64; 10] {
        [
            self.a as u64,
            self.b as u64,
            self.sa as u64,
            self.sb.unwrap_or(16) as u64,
            self.check as u64,
            self.search as u64,
            self.total as u64,
            self.score.to_bits(),
            u64::from(self.fast[0]) + 2 * u64::from(self.fast[1]) + 4 * u64::from(self.transported),
            0x7072_6573_6572_7665,
        ]
    }
}

fn head_score(p: &HullMarginProblem, j: usize) -> Option<f64> {
    let mut score = p.b1[j].abs();
    for i in 0..p.input_dim {
        let radius = finite(p.lower[i].abs().max(p.upper[i].abs()))?;
        score = finite(score + p.w_row(j)[i].abs() * radius)?;
    }
    finite(p.c[j] * score)
}

fn choose(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    ticket: &Ticket,
    old_cost: usize,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Choice> {
    ticket.claim_selection_mode(cuts, b, plan, old_cost, true)?;
    let p = cuts.problem;
    // This producer is entered directly after geometry::plan. Whole owners
    // give disjoint positive supports, including for a saturated exchange.
    if plan
        .components
        .iter()
        .any(|c| c.positive_units.iter().any(|u| *u != 2))
    {
        return None;
    }
    let mut slots = vec![None; plan.components.len()];
    let mut uniform_seed = vec![false; plan.components.len()];
    for (index, c) in plan.components.iter().enumerate() {
        ticket.alive(cuts, b, poll)?;
        uniform_seed[index] = uniform(c);
        let mut best = f64::NEG_INFINITY;
        for (slot, &j) in c.positive.iter().enumerate() {
            let score = head_score(p, j)?;
            if score > best {
                best = score;
                slots[index] = Some(slot);
            }
        }
    }
    let free = ticket.remaining(cuts, b)?;
    let whole_words = resources::words(plan)?;
    let mut best: Option<Choice> = None;
    for a in 0..plan.components.len() {
        let Some(sa) = slots[a] else { continue };
        for other in 0..plan.components.len() {
            ticket.alive(cuts, b, poll)?;
            if a == other {
                continue;
            }
            let ca = &plan.components[a];
            let cb = &plan.components[other];
            if ca.negative.len().max(cb.negative.len()) <= 6 {
                continue;
            }
            let sb = if cb.positive.len() < 16 {
                None
            } else {
                // The reverse orientation supplies half sharing whenever
                // either component has a free slot. Only full/full exchanges.
                if ca.positive.len() != 16 || a > other {
                    continue;
                }
                Some(slots[other]?)
            };
            let Some((check, search, total)) = quote::choice(
                plan,
                p.input_dim,
                a,
                other,
                sb.is_some(),
                [uniform_seed[a], uniform_seed[other]],
                old_cost,
                whole_words,
            ) else {
                continue;
            };
            if total > free {
                continue;
            }
            let mut score = 0.;
            for i in 0..p.input_dim {
                let x = ca.g[i];
                let y = cb.g[i];
                let radius = p.lower[i].abs().max(p.upper[i].abs());
                score = finite(score + (x.abs() + y.abs() - (x + y).abs()).max(0.) * radius)?;
            }
            // Stable source-derived ordering, with one positive head per owner.
            if best.is_none_or(|old| score > old.score) {
                best = Some(Choice {
                    a,
                    b: other,
                    sa,
                    sb,
                    fast: [uniform_seed[a], uniform_seed[other]],
                    transported: false,
                    check,
                    search,
                    total,
                    score,
                });
            }
        }
    }
    let mut choice = best?;
    // Pair selection remains the existing tournament. Only the selected,
    // already funded pair can purchase the uniform transported competitor.
    if ticket.transported_payload && transported::eligible(cuts, plan, choice) {
        if let Some(extra) = quote::transported_extra(plan, p.input_dim, choice.a, choice.b) {
            if let Some(total) = choice.total.checked_add(extra) {
                if total <= free {
                    choice.search = choice.search.checked_add(extra)?;
                    choice.total = total;
                    choice.transported = true;
                }
            }
        }
    }
    // Keep the already funded pair and face competitor. The oblique trial
    // uses their existing maximum scratch allowance, but buys its own work.
    let mut oblique = false;
    let mut oblique_fee = 0;
    if choice.transported {
        if let Some(extra) = quote::oblique_extra(plan, p.input_dim, choice.a, choice.b) {
            if let Some(total) = choice.total.checked_add(extra) {
                if total <= free {
                    choice.search = choice.search.checked_add(extra)?;
                    choice.total = total;
                    oblique = true;
                    oblique_fee = extra;
                }
            }
        }
    }
    let mut u = ticket.state.lock().ok()?;
    if !u.selection || u.choice.is_some() || !ticket.owned(cuts, b) {
        return None;
    }
    u.choice = Some(choice.bits());
    u.oblique_choice = oblique.then_some(choice.bits());
    u.oblique_fee = oblique_fee;
    u.quote = choice.total;
    Some(choice)
}

#[derive(Clone, Copy)]
struct Edit {
    // Existing slot, or the old positive count for an appended receiver.
    slot: usize,
    head: usize,
    units: u8,
    incoming: bool,
}
fn edits(plan: &Plan, choice: Choice) -> (Edit, Edit) {
    let a = &plan.components[choice.a];
    let b = &plan.components[choice.b];
    match choice.sb {
        None => (
            Edit {
                slot: choice.sa,
                head: a.positive[choice.sa],
                units: 1,
                incoming: false,
            },
            Edit {
                slot: b.positive.len(),
                head: a.positive[choice.sa],
                units: 1,
                incoming: true,
            },
        ),
        Some(sb) => (
            Edit {
                slot: choice.sa,
                head: b.positive[sb],
                units: 2,
                incoming: true,
            },
            Edit {
                slot: sb,
                head: a.positive[choice.sa],
                units: 2,
                incoming: true,
            },
        ),
    }
}
fn seed(cuts: &HullRootMipCuts<'_>, c: &Component, e: Edit, mask: usize) -> f64 {
    if e.incoming {
        sanitize_alpha(cuts.multipliers.alpha[e.head])
    } else {
        c.alpha[mask * c.positive.len() + e.slot]
    }
}
fn interpolated(a: f64, b: f64, t: f64) -> Option<f64> {
    finite(a + t * finite(b - a)?)
}

struct Cache {
    // Row [constant, q_0, ..., q_(D-1)] at seed and endpoint-minus-seed.
    start: Vec<f64>,
    delta: Vec<f64>,
    endpoint: Vec<f64>,
    mean: Vec<f64>,
    split: usize,
}
impl Cache {
    fn new(d: usize, ma: usize, mb: usize) -> Option<Self> {
        let m = ma.checked_add(mb)?;
        let n = m.checked_mul(d.checked_add(1)?)?;
        Some(Self {
            start: vec![0.; n],
            delta: vec![0.; n],
            endpoint: vec![0.; m],
            mean: vec![0.; d],
            split: ma,
        })
    }
}
fn uniform(c: &Component) -> bool {
    let p = c.positive.len();
    p == 0
        || c.alpha.chunks_exact(p).all(|row| {
            row.iter()
                .zip(&c.alpha[..p])
                .all(|(x, y)| x.to_bits() == y.to_bits())
        })
}
fn add_head(p: &HullMarginProblem, out: &mut [f64], j: usize, scale: f64) -> Option<()> {
    out[0] = finite(out[0] + scale * p.b1[j])?;
    for i in 0..p.input_dim {
        out[i + 1] = finite(out[i + 1] + scale * p.w_row(j)[i])?;
    }
    Some(())
}
fn common(
    cuts: &HullRootMipCuts<'_>,
    c: &Component,
    e: Edit,
    mask: usize,
    out: &mut [f64],
) -> Option<()> {
    out[0] = 0.;
    for (q, g) in out[1..].iter_mut().zip(&c.g) {
        *q = -*g;
    }
    for (slot, &j) in c.positive.iter().enumerate() {
        if slot != e.slot {
            add_head(
                cuts.problem,
                out,
                j,
                finite(cuts.problem.c[j] * c.alpha[mask * c.positive.len() + slot])?,
            )?;
        }
    }
    let coeff = finite(cuts.problem.c[e.head] * (f64::from(e.units) * 0.5))?;
    add_head(
        cuts.problem,
        out,
        e.head,
        finite(coeff * seed(cuts, c, e, mask))?,
    )
}
fn negatives(p: &HullMarginProblem, c: &Component, mask: usize, out: &mut [f64]) -> Option<()> {
    for (bit, &j) in c.negative.iter().enumerate() {
        if mask & (1usize << bit) != 0 {
            add_head(p, out, j, p.c[j])?;
        }
    }
    Some(())
}
fn build_cache(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    choice: Choice,
    b: &Budget,
    ticket: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Cache> {
    let p = cuts.problem;
    let a = &plan.components[choice.a];
    let z = &plan.components[choice.b];
    let (ea, ez) = edits(plan, choice);
    let mut cache = Cache::new(
        p.input_dim,
        1usize << a.negative.len(),
        1usize << z.negative.len(),
    )?;
    for i in 0..p.input_dim {
        cache.mean[i] = finite(0.5 * a.g[i] + 0.5 * z.g[i])?;
    }
    let width = p.input_dim + 1;
    let mut work = vec![0.; width];
    let mut base = vec![0.; width];
    for (c, e, offset, fast) in [
        (a, ea, 0, choice.fast[0]),
        (z, ez, cache.split, choice.fast[1]),
    ] {
        ticket.alive(cuts, b, poll)?;
        if uniform(c) != fast {
            return None;
        }
        if fast {
            common(cuts, c, e, 0, &mut base)?;
        }
        let mut previous = 0usize;
        let masks = 1usize << c.negative.len();
        for index in 0..masks {
            ticket.alive(cuts, b, poll)?;
            let mask = if fast { index ^ (index >> 1) } else { index };
            if !fast {
                common(cuts, c, e, mask, &mut work)?;
                negatives(p, c, mask, &mut work)?;
            } else if index % 16 == 0 {
                work.copy_from_slice(&base);
                negatives(p, c, mask, &mut work)?;
            } else {
                let changed = mask ^ previous;
                let bit = changed.trailing_zeros() as usize;
                let j = c.negative[bit];
                let sign = if mask & changed == 0 { -1. } else { 1. };
                add_head(p, &mut work, j, finite(sign * p.c[j])?)?;
            }
            previous = mask;
            let row = (offset + mask) * width;
            cache.start[row..row + width].copy_from_slice(&work);
        }
    }
    Some(cache)
}

fn nominal(p: &HullMarginProblem, row: &[f64], direction: &[f64], amount: f64) -> Option<f64> {
    let mut value = finite(row[0] + amount * direction[0])?;
    for i in 0..p.input_dim {
        let q = finite(row[i + 1] + amount * direction[i + 1])?;
        value = finite(value + (q * p.lower[i]).min(q * p.upper[i]))?;
    }
    Some(value)
}
fn endpoints(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    choice: Choice,
    cache: &mut Cache,
    b: &Budget,
    ticket: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    let p = cuts.problem;
    let width = p.input_dim + 1;
    let (ea, eb) = edits(plan, choice);
    let mut v = vec![0.; width];
    let mut residual = vec![0.; width];
    for (index, e, offset) in [(choice.a, ea, 0), (choice.b, eb, cache.split)] {
        let c = &plan.components[index];
        v.fill(0.);
        let coefficient = finite(p.c[e.head] * (f64::from(e.units) * 0.5))?;
        add_head(p, &mut v, e.head, coefficient)?;
        let mut dominant = 0usize;
        let mut importance = 0.;
        for i in 0..p.input_dim {
            let score = finite(v[i + 1].abs() * finite(p.upper[i] - p.lower[i])?)?;
            if score > importance {
                dominant = i;
                importance = score;
            }
        }
        for mask in 0..1usize << c.negative.len() {
            ticket.alive(cuts, b, poll)?;
            let row = (offset + mask) * width;
            let old = seed(cuts, c, e, mask);
            residual[0] = finite(cache.start[row] - old * v[0])?;
            for i in 0..p.input_dim {
                residual[i + 1] =
                    finite(cache.start[row + i + 1] - old * v[i + 1] + c.g[i] - cache.mean[i])?;
            }
            let crossing = if importance > 0. {
                finite((-residual[dominant + 1] / v[dominant + 1]).clamp(0., 1.))?
            } else {
                old
            };
            let mut selected = old;
            let mut best = nominal(p, &residual, &v, old)?;
            for candidate in [0., 1., crossing] {
                let score = nominal(p, &residual, &v, candidate)?;
                if score > best {
                    selected = candidate;
                    best = score;
                }
            }
            cache.endpoint[offset + mask] = selected;
            cache.delta[row] = finite((selected - old) * v[0])?;
            for i in 0..p.input_dim {
                cache.delta[row + i + 1] =
                    finite((selected - old) * v[i + 1] + c.g[i] - cache.mean[i])?;
            }
            note(b, |r| {
                r.line_mask_attempts += 1;
                r.line_masks_completed += 1;
                r.line_masks_improved += usize::from(selected != old);
            });
        }
    }
    Some(())
}
fn choose_line(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    choice: Choice,
    cache: &Cache,
    b: &Budget,
    ticket: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(f64, f64)> {
    let p = cuts.problem;
    let width = p.input_dim + 1;
    let a = &plan.components[choice.a];
    let z = &plan.components[choice.b];
    let mut global = vec![0.; width];
    let mut delta = vec![0.; width];
    for c in &plan.components {
        for i in 0..p.input_dim {
            global[i + 1] = finite(global[i + 1] + c.g[i])?;
        }
    }
    for &j in &plan.leftover {
        add_head(
            p,
            &mut global,
            j,
            finite(p.c[j] * sanitize_alpha(cuts.multipliers.alpha[j]))?,
        )?;
    }
    for i in 0..p.input_dim {
        delta[i + 1] = finite(finite(cache.mean[i] - a.g[i])? + finite(cache.mean[i] - z.g[i])?)?;
    }
    let mut chosen = 0.;
    let mut best = f64::NEG_INFINITY;
    for t in [0., 0.5, 1.] {
        let mut score = nominal(p, &global, &delta, t)?;
        for (begin, end) in [(0, cache.split), (cache.split, cache.endpoint.len())] {
            let mut lower = f64::INFINITY;
            for mask in begin..end {
                ticket.alive(cuts, b, poll)?;
                let row = mask * width;
                lower = lower.min(nominal(
                    p,
                    &cache.start[row..row + width],
                    &cache.delta[row..row + width],
                    t,
                )?);
            }
            score = finite(score + lower)?;
        }
        if score > best {
            chosen = t;
            best = score;
        }
    }
    Some((chosen, best))
}

fn clone_part(c: &Component) -> Component {
    Component {
        negative: c.negative.clone(),
        positive: c.positive.clone(),
        positive_units: c.positive_units.clone(),
        g: c.g.clone(),
        alpha: c.alpha.clone(),
    }
}
fn rebuild(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    choice: Choice,
    endpoint: &[f64],
    mean: &[f64],
    targets: Option<&[f64]>,
    t: f64,
    b: &Budget,
    ticket: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Plan> {
    let (ea, eb) = edits(plan, choice);
    let split = 1usize << plan.components[choice.a].negative.len();
    let mut result = Plan {
        components: Vec::with_capacity(plan.components.len()),
        leftover: plan.leftover.clone(),
    };
    for (index, c) in plan.components.iter().enumerate() {
        ticket.alive(cuts, b, poll)?;
        let edited = if index == choice.a {
            Some((ea, 0))
        } else if index == choice.b {
            Some((eb, split))
        } else {
            None
        };
        let Some((edit, offset)) = edited else {
            result.components.push(clone_part(c));
            continue;
        };
        let count = c.positive.len().max(edit.slot + 1);
        let mut positive = vec![0usize; count];
        let mut units = vec![2u8; count];
        positive[..c.positive.len()].copy_from_slice(&c.positive);
        positive[edit.slot] = edit.head;
        units[edit.slot] = edit.units;
        let mut alpha = vec![0.; (1usize << c.negative.len()).checked_mul(count)?];
        for mask in 0..1usize << c.negative.len() {
            ticket.alive(cuts, b, poll)?;
            for slot in 0..count {
                alpha[mask * count + slot] = if slot == edit.slot {
                    if targets.is_some() {
                        finite(endpoint[offset + mask])?.clamp(0., 1.)
                    } else {
                        interpolated(seed(cuts, c, edit, mask), endpoint[offset + mask], t)?
                            .clamp(0., 1.)
                    }
                } else {
                    c.alpha[mask * c.positive.len() + slot]
                };
            }
        }
        let mut g = vec![0.; c.g.len()];
        for i in 0..g.len() {
            g[i] = if let Some(targets) = targets {
                targets[if index == choice.a { i } else { c.g.len() + i }]
            } else {
                interpolated(c.g[i], mean[i], t)?
            };
        }
        result.components.push(Component {
            negative: c.negative.clone(),
            positive,
            positive_units: units,
            g,
            alpha,
        });
    }
    quote::new_capacity_bound(plan)?.checked_sub(capacity(&result)?)?;
    Some(result)
}

pub(super) fn run(
    cuts: &HullRootMipCuts<'_>,
    plan: &mut Plan,
    baseline: &mut certificate::Checked,
    cost: &mut usize,
    floor: f64,
    b: &mut Budget,
    ticket: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> bool {
    let Some(choice) = choose(cuts, plan, b, ticket, *cost, poll) else {
        stop(b, "no-funded-preserved-pair");
        return true;
    };
    eprintln!(
        "[mip-root-transport] enabled={} payload={} exchange={} original_positive_a={} original_positive_b={} head_a={} head_b={:?}",
        usize::from(choice.transported), usize::from(ticket.transported_payload),
        usize::from(choice.sb.is_some()), plan.components[choice.a].positive.len(),
        plan.components[choice.b].positive.len(), plan.components[choice.a].positive[choice.sa],
        choice.sb.map(|slot| plan.components[choice.b].positive[slot]),
    );
    if b.reserve_at_least((*cost).max(choice.check)).is_none()
        || ticket.pay(cuts, b, choice.total).is_none()
    {
        stop(b, "preserved-whole-quote-refused");
        return true;
    }
    {
        let Some(mut u) = ticket.state.lock().ok() else {
            return true;
        };
        if u.build || !u.paid_choice || u.choice != Some(choice.bits()) {
            stop(b, "preserved-build-permit");
            return true;
        }
        u.build = true;
        u.selected_search = true;
        u.recipient_search = true;
    }
    if !ticket.matches_plan(plan, *cost) || ticket.alive(cuts, b, poll).is_none() {
        stop(b, "preserved-parent-seal-refused");
        return true;
    }
    note(b, |r| {
        r.left = Some(choice.a);
        r.right = Some(choice.b);
        r.parent_count = 2;
        r.parent_indices[..2].copy_from_slice(&[choice.a, choice.b]);
        r.selected_negatives = plan.components[choice.a].negative.len();
        r.tail_negatives = 0;
        r.funded_coordinates = 2;
        r.quoted_check_work = choice.check;
        r.quoted_search_work = choice.search;
    });
    let outcome = (|| {
        let mut cache = build_cache(cuts, plan, choice, b, ticket, poll)?;
        endpoints(cuts, plan, choice, &mut cache, b, ticket, poll)?;
        let (t, score) = choose_line(cuts, plan, choice, &cache, b, ticket, poll)?;
        let proposal = if choice.transported {
            let mut cancelled = false;
            let mut permit = || {
                let keep = poll();
                cancelled |= !keep;
                keep
            };
            let proposal =
                transported::propose(cuts, plan, choice, &mut cache, b, ticket, &mut permit);
            if cancelled {
                return None;
            }
            // A numerical refusal may retain the canonical proposal. A
            // cancelled/expired/foreign ticket may not continue to a check.
            ticket.alive(cuts, b, poll)?;
            eprintln!(
                "[mip-root-transport-score] canonical={:?} transported={:?} axis={:?} stored_changes={:?}",
                score, proposal.as_ref().map(|p| p.score),
                proposal.as_ref().and_then(|p| p.axis),
                proposal.as_ref().map(|p| p.stored_changes),
            );
            proposal
        } else {
            None
        };
        // Deliberate lifetime barrier: no full Plan clone coexists with caches.
        let Cache {
            start,
            delta,
            endpoint,
            mean,
            ..
        } = cache;
        drop(start);
        drop(delta);
        if let Some(proposal) = proposal {
            if proposal.score > score {
                // No discarded endpoint/mean buffers coexist with the Plan.
                drop(endpoint);
                drop(mean);
                return rebuild(
                    cuts,
                    plan,
                    choice,
                    &proposal.endpoint,
                    &[],
                    Some(&proposal.targets),
                    1.,
                    b,
                    ticket,
                    poll,
                );
            }
            drop(proposal);
        }
        rebuild(
            cuts, plan, choice, &endpoint, &mean, None, t, b, ticket, poll,
        )
    })();
    let Some(candidate) = outcome else {
        stop(b, "preserved-search-refused");
        return true;
    };
    let Some(new_cost) = candidate.cost(cuts.problem) else {
        stop(b, "preserved-whole-cost-refused");
        return true;
    };
    if new_cost > choice.check || ticket.claim_checker_mode(cuts, b, true).is_none() {
        stop(b, "preserved-checker-permit");
        return true;
    }
    let Some(checked) = certificate::check(cuts, &candidate, b, ticket.end, poll) else {
        stop(b, "preserved-whole-source-refused");
        return true;
    };
    note(b, |r| {
        r.completed_supports += 1;
        r.actual_check_work = new_cost;
        r.new_masks = candidate.masks().unwrap_or(0);
        r.positive_heads = candidate.components[choice.a].positive.len();
        r.recipient_positive_heads = candidate.components[choice.b].positive.len();
        r.optimized_coordinates = 2;
        r.proposed_lower = Some(checked.root_lower);
    });
    if ticket.alive(cuts, b, poll).is_none() || require_hull_environment().is_err() {
        stop(b, "preserved-completion-barrier");
        return true;
    }
    if checked.root_lower > baseline.root_lower.max(floor) {
        *plan = candidate;
        *baseline = checked;
        *cost = new_cost;
        note(b, |r| {
            r.selected = true;
            r.after_global_lower = Some(baseline.root_lower);
            r.stop_reason = "preserved-root-improved";
        });
    } else {
        stop(b, "preserved-baseline-retained");
    }
    true
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_preserved_tests.rs"]
mod tests;
