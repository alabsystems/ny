// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Multibit supports from the losing allocation's existing final Source pass.
//! The full table certifies the serialized row; captured activity is only advice.
use super::*;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_mass_cube_retention_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug)]
pub(super) struct Quote {
    components: usize,
    negatives: usize,
    masks: usize,
    visits: usize,
    terms: usize,
    work: usize,
    bytes: usize,
}

pub(super) fn upgrade(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    minimum_advice: usize,
    permit: &mut Permit,
) -> Option<()> {
    if permit.conditional.is_none() || !owned(permit, cuts, b) || permit.final_paid {
        return None;
    }
    let (d, h, _) = b.dimensions;
    let components = plan.components.len();
    let setup = h
        .checked_add(d)?
        .checked_add(components)?
        .checked_add(2)?
        .checked_mul(32)?;
    let protected = permit
        .quote
        .cost
        .checked_mul(2)?
        .checked_add(minimum_advice)?;
    {
        let mut c = b.state.lock().ok()?;
        if c.portfolio.report.cube.attempted || c.replay_attempts != 0 {
            return None;
        }
        c.portfolio.report.cube.attempted = true;
        let work = c.work.checked_add(setup)?;
        if work
            .checked_add(c.reserved)?
            .checked_add(c.component_reserved)?
            .checked_add(protected)?
            > MAX_WORK
            || !live(b.search_end)
        {
            return None;
        }
        c.work = work;
        c.portfolio.report.cube.setup_work = setup;
        c.portfolio.report.cube.work_units = setup;
    }
    let (mut negatives, mut masks, mut visits, mut square) = (0usize, 0usize, 0usize, 0usize);
    for item in &plan.components {
        if !live(b.search_end) {
            return None;
        }
        let k = item.negative.len();
        if !(1..=8).contains(&k) {
            return None;
        }
        negatives = negatives.checked_add(k)?;
        masks = masks.checked_add(1usize << k)?;
        visits = visits.checked_add(k.checked_mul(1usize << k)?)?;
        square = square.checked_add(k.checked_add(1)?.checked_pow(2)?)?;
    }
    if masks > MAX_MASKS
        || size_of::<Collector>() > 256
        || size_of::<Recipe>() > 256
        || size_of::<Permit>() > 512
        || size_of::<State>() > 4096
        || size_of::<Option<[f64; 8]>>() > 72
    {
        return None;
    }
    let terms = permit.quote.terms.checked_add(negatives)?;
    let span = terms
        .checked_add(negatives.checked_mul(2)?)?
        .checked_add(h)?
        .checked_add(d)?
        .checked_add(components)?
        .checked_add(1)?;
    // Includes table retention/seal/shape walks, bounded proposal sorting,
    // both complete directed repair walks, mapping, and the additional headers.
    let work = visits
        .checked_mul(64)?
        .checked_add(masks.checked_mul(32)?)?
        .checked_add(square.checked_mul(64)?)?
        .checked_add(span.checked_mul(128)?)?
        .checked_add(4096)?;
    let bytes = masks
        .checked_mul(32)?
        .checked_add(terms.checked_mul(64)?)?
        .checked_add(negatives.checked_mul(128)?)?
        .checked_add(components.checked_mul(768)?)?
        .checked_add(h.checked_mul(32)?)?
        .checked_add(d.checked_add(1)?.checked_mul(64)?)?
        .checked_add(4096)?;
    let quote = Quote {
        components,
        negatives,
        masks,
        visits,
        terms,
        work,
        bytes,
    };
    let rows = permit.quote.rows.checked_add(components)?;
    let mut c = b.state.lock().ok()?;
    let all_rows = phase::portfolio_header_upgrade(&c, b, components)?;
    let work = c.work.checked_add(quote.work.checked_sub(setup)?)?;
    let shared = c.shared_bytes.checked_add(quote.bytes)?;
    let total_bytes = b
        .admitted_bytes
        .checked_add(shared)?
        .checked_add(c.direction.cache_bytes)?
        .checked_add(c.child.cache_bytes)?
        .checked_add(c.component_rows.admitted_bytes)?;
    if work
        .checked_add(c.reserved)?
        .checked_add(c.component_reserved)?
        .checked_add(protected)?
        > MAX_WORK
        || total_bytes > MAX_BYTES
        || !live(b.search_end)
    {
        return None;
    }
    phase::apply_portfolio_header_upgrade(&mut c, all_rows)?;
    c.work = work;
    c.shared_bytes = shared;
    let r = &mut c.portfolio.report.cube;
    r.admitted = true;
    r.quoted_masks = masks;
    r.quoted_rows = components;
    r.quoted_terms = terms;
    r.work_units = quote.work;
    r.admitted_bytes = quote.bytes;
    r.stop_reason = "prepaid";
    permit.quote.rows = rows;
    permit.cube = Some(quote);
    Some(())
}

pub(super) struct Collector {
    quote: Quote,
    tables: Vec<Vec<f64>>,
    activity: Vec<Option<[f64; 8]>>,
    negatives: usize,
    masks: usize,
    visits: usize,
    valid: bool,
}
pub(super) fn collector(q: Quote, plan: &Plan, b: &Budget) -> Option<Box<Collector>> {
    if plan.components.len() != q.components || !live(b.search_end) {
        return None;
    }
    {
        let mut c = b.state.lock().ok()?;
        let r = &mut c.portfolio.report.cube;
        if !r.admitted || r.collection_attempts != 0 {
            return None;
        }
        r.collection_attempts = 1;
        r.activity_transfer_attempts = 1;
        r.stop_reason = "collecting";
    }
    // This copies only the already transferred advice. It never reopens the
    // captured LP account, and its N+C traversal is included in this quote.
    let activity = phase::cube::portfolio_activity(plan, b)?;
    let tables = Vec::with_capacity(q.components);
    if activity.capacity() > 2 * q.components || tables.capacity() > 2 * q.components {
        return None;
    }
    let mut c = b.state.lock().ok()?;
    let r = &mut c.portfolio.report.cube;
    r.transferred_activity_values = plan
        .components
        .iter()
        .zip(&activity)
        .filter(|(_, a)| a.is_some())
        .map(|(p, _)| p.negative.len())
        .sum();
    r.activity_values = r.transferred_activity_values;
    Some(Box::new(Collector {
        quote: q,
        tables,
        activity,
        negatives: 0,
        masks: 0,
        visits: 0,
        valid: true,
    }))
}
impl Collector {
    pub(super) fn push(&mut self, component: &Component, values: Option<Vec<f64>>) {
        let k = component.negative.len();
        let Some(values) = values else {
            self.valid = false;
            return;
        };
        if !self.valid
            || !(1..=8).contains(&k)
            || self.tables.len() >= self.quote.components
            || values.len() != 1usize << k
            || values.capacity() > 2 * values.len()
            || values.iter().any(|x| !x.is_finite())
        {
            self.valid = false;
            return;
        }
        self.negatives += k;
        self.masks += values.len();
        self.visits += k * values.len();
        if self.negatives > self.quote.negatives
            || self.masks > self.quote.masks
            || self.visits > self.quote.visits
        {
            self.valid = false;
            return;
        }
        self.tables.push(values);
    }
    pub(super) fn finish(self, planes: &[Plane]) -> Option<Box<Recipe>> {
        if !self.valid
            || self.negatives != self.quote.negatives
            || self.masks != self.quote.masks
            || self.visits != self.quote.visits
        {
            return None;
        }
        let mut out = Recipe {
            quote: self.quote,
            tables: self.tables,
            activity: self.activity,
            seal: Vec::new(),
        };
        if !out.shape(planes) {
            return None;
        }
        // Tables, advice tags and actual planes need at most M+2T+13C+5
        // words. Reserve this exact nominal bound before copying any bits.
        let capacity = out
            .quote
            .masks
            .checked_add(out.quote.terms.checked_mul(2)?)?
            .checked_add(out.quote.components.checked_mul(13)?)?
            .checked_add(5)?;
        let mut seal = Vec::with_capacity(capacity);
        if seal.capacity() > capacity {
            return None;
        }
        seal.extend(out.bits(planes));
        if seal.len() > capacity {
            return None;
        }
        out.seal = seal;
        Some(Box::new(out))
    }
}

#[derive(Debug)]
pub(super) struct Recipe {
    quote: Quote,
    tables: Vec<Vec<f64>>,
    activity: Vec<Option<[f64; 8]>>,
    seal: Vec<u64>,
}
pub(super) struct Prepared {
    pub rows: Vec<RowSpec>,
    pub heads: usize,
    pub terms: usize,
    pub selectors: usize,
    pub advised: usize,
    pub projected: usize,
}
impl Recipe {
    fn bits<'a>(&'a self, planes: &'a [Plane]) -> impl Iterator<Item = u64> + 'a {
        [
            self.quote.components as u64,
            self.quote.negatives as u64,
            self.quote.masks as u64,
            self.quote.visits as u64,
            self.quote.terms as u64,
        ]
        .into_iter()
        .chain(
            self.tables
                .iter()
                .flat_map(|v| std::iter::once(v.len() as u64).chain(v.iter().map(|x| x.to_bits()))),
        )
        .chain(self.activity.iter().flat_map(|a| {
            std::iter::once(u64::from(a.is_some()))
                .chain(a.iter().flat_map(|v| v.iter().map(|x| x.to_bits())))
        }))
        .chain(planes.iter().flat_map(|p| {
            [
                p.heads.len() as u64,
                p.input.len() as u64,
                p.lower.to_bits(),
            ]
            .into_iter()
            .chain(p.heads.iter().flat_map(|(j, a)| [*j as u64, a.to_bits()]))
            .chain(p.input.iter().map(|x| x.to_bits()))
        }))
    }
    fn shape(&self, planes: &[Plane]) -> bool {
        self.tables.len() == self.quote.components
            && planes.len() == self.tables.len()
            && self.activity.len() == planes.len()
            && self.tables.capacity() <= 2 * self.quote.components
            && self.activity.capacity() <= 2 * self.quote.components
            && planes
                .iter()
                .try_fold(0usize, |n, p| {
                    n.checked_add(p.heads.len().checked_mul(2)?)?
                        .checked_add(p.input.len())
                })
                .is_some_and(|n| n <= 2 * self.quote.terms)
            && self
                .activity
                .iter()
                .flatten()
                .flatten()
                .all(|v| v.is_finite() && (0. ..=1.).contains(v))
            && self.tables.iter().zip(planes).all(|(v, p)| {
                let k = p.heads.iter().take_while(|(_, a)| *a < 0.).count();
                (1..=8).contains(&k)
                    && v.len() == 1usize << k
                    && v.capacity() <= 2 * v.len()
                    && v.iter().all(|x| x.is_finite())
                    && v.iter().copied().fold(f64::INFINITY, f64::min) == p.lower * 0.5
            })
    }
    pub(super) fn prepare(
        &self,
        planes: &[Plane],
        plain: &[RowSpec],
        binding: &binding::SignedPhaseBinding<'_>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Prepared> {
        if !self.shape(planes)
            || plain.len() != planes.len()
            || !self.bits(planes).eq(self.seal.iter().copied())
        {
            return None;
        }
        let mut out = Prepared {
            rows: Vec::with_capacity(self.quote.components),
            heads: 0,
            terms: 0,
            selectors: 0,
            advised: 0,
            projected: 0,
        };
        if out.rows.capacity() > self.quote.components {
            return None;
        }
        for (index, ((values, plane), base)) in
            self.tables.iter().zip(planes).zip(plain).enumerate()
        {
            if !live(end) || !poll() {
                return None;
            }
            let k = plane.heads.iter().take_while(|(_, a)| *a < 0.).count();
            if k < 2 {
                continue;
            }
            let mut heads = [0; 8];
            for bit in 0..k {
                heads[bit] = plane.heads[bit].0;
            }
            let activity = self.activity[index].as_ref().map(|v| &v[..k]);
            let weights = phase::cube::support_weights(values, &heads[..k], activity)?;
            let (mut encoded, mut columns, mut used) = ([0.; 8], [None; 8], 0usize);
            for bit in 0..k {
                columns[bit] = binding.column(heads[bit]);
                if columns[bit].is_some() {
                    encoded[bit] = finite(-2. * weights[bit])?;
                    used += usize::from(encoded[bit] != 0.);
                }
            }
            if used < 2 {
                continue;
            }
            let lower = phase::cube::repair(values, &encoded[..k], end, poll)?;
            let needed = base.coeffs.len().checked_add(used)?;
            let mut coeffs = Vec::with_capacity(needed);
            if coeffs.capacity() > needed {
                return None;
            }
            coeffs.extend_from_slice(&base.coeffs);
            for bit in 0..k {
                if encoded[bit] != 0. {
                    coeffs.push((columns[bit]?.0, encoded[bit]));
                }
            }
            out.terms = out.terms.checked_add(coeffs.len())?;
            out.heads = out.heads.checked_add(plane.heads.len())?;
            out.selectors = out.selectors.checked_add(used)?;
            if out.terms > self.quote.terms || out.rows.len() >= self.quote.components {
                return None;
            }
            out.rows.push(RowSpec {
                lb: lower,
                ub: f64::INFINITY,
                coeffs,
            });
            out.advised += usize::from(activity.is_some());
            out.projected += usize::from(columns[..k].iter().any(|c| c.is_none()));
        }
        require_hull_environment().ok()?;
        (live(end) && poll()).then_some(out)
    }
}
