// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One selector lift per component of a strictly losing checked allocation.
//! The existing final mass pass is the only producer of conditional minima.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Quote {
    components: usize,
    negatives: usize,
    visits: usize,
    terms: usize,
    work: usize,
    bytes: usize,
}

/// Additive, once-only upgrade; every refusal retains the ordinary permit.
pub(super) fn upgrade(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    minimum_advice: usize,
    permit: &mut Permit,
) -> Option<()> {
    if !b.portfolio_phase_enabled || !owned(permit, cuts, b) || permit.final_paid {
        return None;
    }
    {
        let mut c = b.state.lock().ok()?;
        if c.portfolio.report.phase_attempted || c.replay_attempts != 0 {
            return None;
        }
        c.portfolio.report.phase_attempted = true;
    }
    let components = plan.components.len();
    let (d, h, _) = b.dimensions;
    let setup = h
        .checked_add(d)?
        .checked_add(components)?
        .checked_add(cuts.len())?
        .checked_add(2)?
        .checked_mul(32)?;
    let protected = permit
        .quote
        .cost
        .checked_mul(2)?
        .checked_add(minimum_advice)?;
    {
        let mut c = b.state.lock().ok()?;
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
        c.portfolio.report.phase_work_units = setup;
    }
    let mut negatives = 0usize;
    let mut visits = 0usize;
    for item in &plan.components {
        let k = item.negative.len();
        if !(1..=8).contains(&k) {
            return None;
        }
        negatives = negatives.checked_add(k)?;
        visits = visits.checked_add(k.checked_mul(1usize << k)?)?;
    }
    let terms = permit.quote.terms.checked_add(components)?;
    let span = terms
        .checked_add(components.checked_mul(3)?)?
        .checked_add(h)?
        .checked_add(d)?
        .checked_add(1)?;
    let work = visits
        .checked_mul(32)?
        .checked_add(span.checked_mul(128)?)?
        .checked_add(4096)?;
    let bytes = terms
        .checked_mul(64)?
        .checked_add(negatives.checked_mul(64)?)?
        .checked_add(components.checked_mul(512)?)?
        .checked_add(4096)?;
    if size_of::<Collector>() > 256
        || size_of::<Recipe>() > 128
        || size_of::<Permit>() > 512
        || size_of::<State>() > 4096
    {
        return None;
    }
    let quote = Quote {
        components,
        negatives,
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
    // Both bounds refer to the same eventual allocation, paid exactly once.
    // No fallible operation follows the metadata update within this lock.
    phase::apply_portfolio_header_upgrade(&mut c, all_rows)?;
    c.work = work;
    c.shared_bytes = shared;
    c.portfolio.report.phase_admitted = true;
    c.portfolio.report.phase_work_units = quote.work;
    c.portfolio.report.phase_admitted_bytes = quote.bytes;
    permit.quote.rows = rows;
    permit.conditional = Some(quote);
    Some(())
}

/// Opaque request/state. Construction requires the paid final-pass permit.
/// The certificate kernel may fill it but cannot manufacture its authority.
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Collector {
    quote: Quote,
    values: Vec<Vec<[f64; 2]>>,
    negatives: usize,
    visits: usize,
    valid: bool,
    cube: Option<Box<cube::Collector>>,
}
pub(super) fn collector(
    permit: &Permit,
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
) -> Option<Collector> {
    let quote = permit.conditional?;
    if !owned(permit, cuts, b)
        || !permit.final_paid
        || quote.components != plan.components.len()
        || layout(plan, b)? != (permit.quote.components, permit.quote.terms)
        || phase::collecting_plan(b, plan)
        || !live(b.search_end)
    {
        return None;
    }
    Some(Collector {
        quote,
        values: Vec::with_capacity(quote.components),
        negatives: 0,
        visits: 0,
        valid: true,
        cube: permit.cube.and_then(|q| cube::collector(q, plan, b)),
    })
}
impl Collector {
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn wants_table(&self) -> bool {
        self.cube.is_some()
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn push(
        &mut self,
        component: &Component,
        pair: Option<Vec<[f64; 2]>>,
        coverage: &[[usize; 2]; 8],
        table: Option<Vec<f64>>,
    ) {
        if let Some(cube) = &mut self.cube {
            cube.push(component, table);
        }
        let k = component.negative.len();
        let Some(pair) = pair else {
            self.valid = false;
            return;
        };
        if !self.valid
            || !(1..=8).contains(&k)
            || self.values.len() >= self.quote.components
            || pair.len() != k
            || pair.iter().flatten().any(|v| !v.is_finite())
            || coverage[..k].iter().any(|v| *v != [(1usize << k) / 2; 2])
        {
            self.valid = false;
            return;
        }
        let Some(negatives) = self.negatives.checked_add(k) else {
            self.valid = false;
            return;
        };
        let Some(visits) = self.visits.checked_add(k * (1usize << k)) else {
            self.valid = false;
            return;
        };
        self.negatives = negatives;
        self.visits = visits;
        if self.negatives > self.quote.negatives || self.visits > self.quote.visits {
            self.valid = false;
            return;
        }
        self.values.push(pair);
    }
    pub(super) fn finish(self, planes: &[Plane]) -> Option<Recipe> {
        if !self.valid
            || self.values.len() != self.quote.components
            || planes.len() != self.values.len()
            || self.negatives != self.quote.negatives
            || self.visits != self.quote.visits
        {
            return None;
        }
        let mut recipe = Recipe {
            quote: self.quote,
            values: self.values,
            seal: Vec::new(),
            cube: self.cube.and_then(|c| c.finish(planes)),
        };
        if !recipe.shape(planes) {
            return None;
        }
        recipe.seal = recipe.bits(planes).collect();
        Some(recipe)
    }
}

#[derive(Debug)]
pub(super) struct Recipe {
    quote: Quote,
    values: Vec<Vec<[f64; 2]>>,
    seal: Vec<u64>,
    cube: Option<Box<cube::Recipe>>,
}
impl Recipe {
    #[cfg(test)]
    pub(super) fn cube_for_test(&mut self) -> &mut cube::Recipe {
        self.cube.as_mut().unwrap()
    }
    pub(super) fn has_cube(&self) -> bool {
        self.cube.is_some()
    }
    pub(super) fn prepare_cube(
        &self,
        planes: &[Plane],
        plain: &[RowSpec],
        binding: &binding::SignedPhaseBinding<'_>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<cube::Prepared> {
        if !self.shape(planes) || !self.bits(planes).eq(self.seal.iter().copied()) {
            return None;
        }
        self.cube
            .as_ref()?
            .prepare(planes, plain, binding, end, poll)
    }
    fn bits<'a>(&'a self, planes: &'a [Plane]) -> impl Iterator<Item = u64> + 'a {
        [
            self.values.len() as u64,
            self.quote.negatives as u64,
            self.quote.visits as u64,
            self.quote.terms as u64,
        ]
        .into_iter()
        .chain(self.values.iter().flat_map(|v| {
            [v.len() as u64]
                .into_iter()
                .chain(v.iter().flat_map(|pair| pair.iter().map(|x| x.to_bits())))
        }))
        .chain(planes.iter().flat_map(|p| {
            [
                p.heads.len() as u64,
                p.input.len() as u64,
                p.lower.to_bits(),
            ]
            .into_iter()
            .chain(p.heads.iter().flat_map(|(j, a)| [*j as u64, a.to_bits()]))
            .chain(p.input.iter().map(|v| v.to_bits()))
        }))
    }
    fn shape(&self, planes: &[Plane]) -> bool {
        self.values.len() == planes.len()
            && self.values.len() == self.quote.components
            && self.values.iter().zip(planes).all(|(pairs, plane)| {
                let k = pairs.len();
                (1..=8).contains(&k)
                    && k <= plane.heads.len()
                    && plane.heads[..k].iter().all(|(_, a)| *a < 0.)
                    && plane.heads[k..].iter().all(|(_, a)| *a > 0.)
                    && pairs.iter().all(|v| {
                        v.iter().all(|x| x.is_finite()) && v[0].min(v[1]) == plane.lower * 0.5
                    })
            })
    }
    /// Numeric refusal keeps every ordinary portfolio row. Cancellation is
    /// separately latched by the caller and aborts its original transaction.
    pub(super) fn prepare(
        &self,
        planes: &[Plane],
        plain: &[RowSpec],
        binding: &binding::SignedPhaseBinding<'_>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<(Vec<RowSpec>, usize)> {
        if !self.shape(planes)
            || plain.len() != planes.len()
            || !self.bits(planes).eq(self.seal.iter().copied())
        {
            return None;
        }
        let mut rows = Vec::with_capacity(self.quote.components);
        let mut terms = 0usize;
        let mut heads = 0usize;
        for ((pairs, plane), base) in self.values.iter().zip(planes).zip(plain) {
            if !live(end) || !poll() {
                return None;
            }
            let mut best: Option<(f64, usize, usize, f64, f64)> = None;
            for (bit, &pair) in pairs.iter().enumerate() {
                if !live(end) || !poll() {
                    return None;
                }
                let head = plane.heads[bit].0;
                let Some(column) = binding.column(head) else {
                    continue;
                };
                let Some((gap, coefficient, lower)) = phase::plane(plane.lower * 0.5, pair) else {
                    continue;
                };
                if gap <= 0. {
                    continue;
                }
                if best.is_none_or(|old| gap > old.0 || (gap == old.0 && head < old.1)) {
                    best = Some((gap, head, column.0, coefficient, lower));
                }
            }
            if let Some((_, _, column, coefficient, lower)) = best {
                let mut coeffs = Vec::with_capacity(base.coeffs.len().checked_add(1)?);
                coeffs.extend_from_slice(&base.coeffs);
                coeffs.push((column, coefficient));
                terms = terms.checked_add(coeffs.len())?;
                heads = heads.checked_add(plane.heads.len())?;
                if terms > self.quote.terms {
                    return None;
                }
                rows.push(RowSpec {
                    lb: lower,
                    ub: f64::INFINITY,
                    coeffs,
                });
            }
        }
        (live(end) && poll()).then_some((rows, heads))
    }
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_mass_conditional_retention_tests.rs"]
mod tests;
