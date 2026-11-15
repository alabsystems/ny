// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Optional root-only closed-phase envelope. Nominal advice is never authority.
use super::*;

const SETUP: usize = 4096;
const HEADER_BYTES: usize = 2048;

#[derive(Debug, Clone, Copy)]
pub(super) struct Quote {
    pub d: usize,
    pub bits: usize,
    pub axes: usize,
    pub masks: usize,
    pub work: usize,
    pub bytes: usize,
}
fn quote(
    d: usize,
    bits: usize,
    axes: usize,
    parent_axes: usize,
    components: usize,
    all_masks: usize,
) -> Option<Quote> {
    if !(1..=32).contains(&d)
        || !(1..=8).contains(&bits)
        || axes == 0
        || axes > parent_axes
        || parent_axes > 7
        || parent_axes > d
        || !(1..=128).contains(&components)
        || all_masks > MAX_MASKS
    {
        return None;
    }
    let masks = 1usize << bits;
    if all_masks < masks {
        return None;
    }
    let z = 1usize << axes;
    let capture = SETUP.checked_add(32usize.checked_mul(masks)?)?;
    let reprice = masks.checked_mul(128usize.checked_mul(d + 1)?.checked_add(128)?)?;
    let collect = masks
        .checked_mul(
            8usize
                .checked_mul(d)?
                .checked_add(64 * axes)?
                .checked_add(64)?,
        )?
        .checked_add(z.checked_mul(24usize.checked_mul(masks)?)?)?
        .checked_add(32usize.checked_mul(z.checked_add(d + axes + 2)?)?)?;
    // Conservative seal/shape allowance, although only the checked root is retained.
    let bind = 64usize.checked_mul(6usize.checked_add(2 * d + 1 + axes + z)?)?;
    let price = 64usize
        .checked_mul(d + parent_axes + components + 1)?
        .checked_add((1usize << parent_axes).checked_mul(8 * bits + 32 + 16 * components)?)?;
    // Both original replay hooks and bounded component metadata. No Source scan.
    let wrapper = 32usize
        .checked_mul(all_masks)?
        .checked_add(128 * components)?;
    let work = capture
        .checked_add(reprice)?
        .checked_add(collect)?
        .checked_add(bind)?
        .checked_add(price)?
        .checked_add(wrapper)?;
    let bytes = 16usize
        .checked_mul(z)?
        .checked_add(32 * d + 16 + 16 * axes + 8192)?
        .checked_add(16usize.checked_mul(masks)?)?
        .checked_add(HEADER_BYTES)?;
    Some(Quote {
        d,
        bits,
        axes,
        masks,
        work,
        bytes,
    })
}

#[derive(Debug)]
pub(super) struct Grant {
    pub quote: Quote,
    pub advice: Option<Box<Advice>>,
}

/// Once-only additive admission, after every existing paid capability.
pub(in crate::single_hidden_hull::mip_cuts::signed) fn prepay(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
) -> Option<()> {
    let mut counts = b.state.lock().ok()?;
    if std::mem::replace(&mut counts.corner.ray_attempted, true) {
        return None;
    }
    if b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.authority > cuts.deadline
        || !live(b.search_end)
        || counts.replay_attempts != 0
        || counts.source_replays != 0
    {
        return None;
    }
    let grant = counts.corner.grant.as_ref()?;
    if grant.stage != 0 || grant.root_ray.is_some() {
        return None;
    }
    let parent_axes = grant.quote.k;
    let bits = phase::cube::paid_ray_bits(&counts, b)?;
    let work = counts.work.checked_add(SETUP)?;
    let child_work = counts.child.work_units.checked_add(SETUP)?;
    if work
        .checked_add(counts.reserved)?
        .checked_add(counts.component_reserved)?
        > MAX_WORK
    {
        return None;
    }
    counts.work = work;
    counts.child.work_units = child_work;
    let all_masks = plan.masks()?;
    if plan.components.len() > 128
        || plan.components.is_empty()
        || plan.components.iter().any(|c| c.negative.len() > bits)
    {
        return None;
    }
    for axes in (1..=parent_axes).rev() {
        let q = quote(
            cuts.problem.input_dim,
            bits,
            axes,
            parent_axes,
            plan.components.len(),
            all_masks,
        )?;
        let extra = q.work.checked_sub(SETUP)?;
        let next_work = counts.work.checked_add(extra)?;
        let next_child_work = counts.child.work_units.checked_add(extra)?;
        let next_bytes = counts.child.cache_bytes.checked_add(q.bytes)?;
        let total_bytes = b
            .admitted_bytes
            .checked_add(counts.shared_bytes)?
            .checked_add(counts.direction.cache_bytes)?
            .checked_add(counts.component_rows.admitted_bytes)?
            .checked_add(next_bytes)?;
        if next_work
            .checked_add(counts.reserved)?
            .checked_add(counts.component_reserved)?
            > MAX_WORK
            || total_bytes > MAX_BYTES
        {
            continue;
        }
        if !live(b.search_end) {
            return None;
        }
        counts.work = next_work;
        counts.child.work_units = next_child_work;
        counts.child.cache_bytes = next_bytes;
        counts.child.corner_ray_axes = axes;
        counts.corner.grant.as_mut()?.root_ray = Some(Box::new(Grant {
            quote: q,
            advice: None,
        }));
        return Some(());
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    head: usize,
    beta: f64,
}
#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Advice {
    quote: Quote,
    target: usize,
    entries: Vec<Entry>,
}
#[derive(Debug)]
pub(super) struct Capture {
    quote: Quote,
    target: Option<usize>,
    entries: Vec<Entry>,
    failed: bool,
}
impl Capture {
    pub(super) fn new(quote: Quote) -> Option<Self> {
        let entries = Vec::with_capacity(quote.masks);
        if entries.capacity() > quote.masks {
            return None;
        }
        Some(Self {
            quote,
            target: None,
            entries,
            failed: false,
        })
    }
    pub(super) fn push(&mut self, component: usize, mask: usize, advice: Option<(usize, f64)>) {
        if self.failed {
            return;
        }
        let target = *self.target.get_or_insert(component);
        if target != component || mask != self.entries.len() || mask >= self.quote.masks {
            self.failed = true;
            return;
        }
        let (head, beta) = advice.unwrap_or((usize::MAX, 0.));
        self.entries.push(Entry { head, beta });
    }
    pub(super) fn finish(self) -> Option<Box<Advice>> {
        if self.failed || self.entries.len() != self.quote.masks {
            return None;
        }
        Some(Box::new(Advice {
            quote: self.quote,
            target: self.target?,
            entries: self.entries,
        }))
    }
}

#[derive(Debug)]
pub(super) struct Builder {
    advice: Box<Advice>,
    axes: Vec<usize>,
    values: Vec<f64>,
    next: usize,
    failed: bool,
}
impl Builder {
    pub(super) fn new(advice: Box<Advice>, axes: &core::Axes, quote: Quote) -> Option<Self> {
        let a = &advice.quote;
        if a.d != quote.d
            || a.bits != quote.bits
            || a.axes != quote.axes
            || a.masks != quote.masks
            || advice.entries.len() != quote.masks
            || axes.selected().len() < quote.axes
        {
            return None;
        }
        let selected = axes.selected()[..quote.axes].to_vec();
        let values = vec![f64::INFINITY; 1usize << quote.axes];
        if selected.capacity() > quote.axes || values.capacity() > 1usize << quote.axes {
            return None;
        }
        Some(Self {
            advice,
            axes: selected,
            values,
            next: 0,
            failed: false,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push(
        &mut self,
        p: &HullMarginProblem,
        component: &Component,
        index: usize,
        mask: usize,
        coeff: &[I],
        constant: I,
        raw_lower: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        if self.failed {
            return None;
        }
        let result = self.push_inner(p, component, index, mask, coeff, constant, raw_lower, poll);
        if result.is_none() {
            self.failed = true;
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn push_inner(
        &mut self,
        p: &HullMarginProblem,
        component: &Component,
        index: usize,
        mask: usize,
        coeff: &[I],
        constant: I,
        raw_lower: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        if index != self.advice.target {
            return Some(());
        }
        let quote = self.advice.quote;
        if mask != self.next
            || mask >= quote.masks
            || coeff.len() != quote.d
            || p.input_dim != quote.d
            || component.negative.len() != quote.bits
            || !poll()
        {
            return None;
        }
        // Failed advice is beta=0 for the WHOLE mask, never a per-corner maximum.
        let mut adjusted = [I::zero(); core::MAX_D];
        adjusted[..coeff.len()].copy_from_slice(coeff);
        let entry = self.advice.entries[mask];
        let candidate = (|| {
            let bit = component.negative.iter().position(|&h| h == entry.head)?;
            if !p.c.get(entry.head).is_some_and(|c| *c < 0.)
                || !entry.beta.is_finite()
                || (mask & (1usize << bit) == 0 && entry.beta < 0.)
                || (mask & (1usize << bit) != 0 && entry.beta > 0.)
            {
                return None;
            }
            let beta = I::point(entry.beta)?;
            let k = constant.add(beta.scale(*p.b1.get(entry.head)?)?)?;
            for (i, q) in adjusted[..coeff.len()].iter_mut().enumerate() {
                *q = q.add(beta.scale(p.w_row(entry.head)[i])?)?;
            }
            let mut lower = k.lo;
            for (i, q) in adjusted[..coeff.len()].iter().enumerate() {
                lower = finite(add_down(lower, q.minimum(p.lower[i], p.upper[i])?))?;
            }
            Some(lower)
        })();
        let (q, mut value) = match candidate {
            Some(lower) => (&adjusted[..coeff.len()], lower),
            None => (coeff, raw_lower),
        };
        let mut delta = [I::zero(); core::MAX_K];
        for (bit, &axis) in self.axes.iter().enumerate() {
            if !poll() {
                return None;
            }
            value = finite(add_down(
                value,
                -q[axis].minimum(p.lower[axis], p.upper[axis])?,
            ))?;
            let low = q[axis].scale(p.lower[axis])?;
            let high = q[axis].scale(p.upper[axis])?;
            value = finite(add_down(value, low.lo))?;
            delta[bit] = high.sub(low)?;
        }
        for step in 0..self.values.len() {
            if step % 32 == 0 && !poll() {
                return None;
            }
            let corner = step ^ (step >> 1);
            if step != 0 {
                let bit = step.trailing_zeros() as usize;
                let change = if corner & (1usize << bit) != 0 {
                    delta[bit].lo
                } else {
                    -delta[bit].hi
                };
                value = finite(add_down(value, change))?;
            }
            self.values[corner] = self.values[corner].min(value);
        }
        self.next += 1;
        poll().then_some(())
    }
    pub(super) fn finish(self) -> Option<Box<Table>> {
        if self.failed
            || self.next != self.advice.quote.masks
            || self.values.iter().any(|v| !v.is_finite())
        {
            return None;
        }
        Some(Box::new(Table {
            target: self.advice.target,
            axes: self.axes,
            values: self.values,
        }))
    }
}

/// Complete target envelope, no global offset and no independent public authority.
#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Table {
    target: usize,
    axes: Vec<usize>,
    values: Vec<f64>,
}
impl Table {
    pub(super) fn view(&self) -> core::RootReplacement<'_> {
        core::RootReplacement {
            target: self.target,
            axes: &self.axes,
            values: &self.values,
        }
    }
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<Entry>() == 16);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(
    size_of::<Grant>()
        + 2 * size_of::<Capture>()
        + 2 * size_of::<Builder>()
        + 2 * size_of::<Advice>()
        + 2 * size_of::<Table>()
        + 512
        <= HEADER_BYTES
);

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_corner_root_ray_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_corner_root_ray_numeric_tests.rs"]
mod numeric_tests;
