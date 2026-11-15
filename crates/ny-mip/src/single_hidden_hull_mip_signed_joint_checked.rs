// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original all-mask pair replay and private ownership-sealed exact row export.
use super::*;
/// Merge original mass exactly. Initial alpha may use both old masks; the
/// independently checked final alpha and ONE common direction define the row.
pub(super) fn union(
    row: &SignedMajorant,
    p: &HullMarginProblem,
    choice: &resources::Choice,
) -> Option<Component> {
    if choice.left >= choice.right {
        return None;
    }
    let a = row.plan.components.get(choice.left)?;
    let b = row.plan.components.get(choice.right)?;
    let negative: Vec<_> = a.negative.iter().chain(&b.negative).copied().collect();
    if !(2..=8).contains(&negative.len())
        || negative
            .iter()
            .enumerate()
            .any(|(i, j)| negative[..i].contains(j))
    {
        return None;
    }
    let mut positive = Vec::with_capacity(a.positive.len() + b.positive.len());
    let mut positive_units: Vec<u8> = Vec::with_capacity(positive.capacity());
    for c in [a, b] {
        for (&j, &units) in c.positive.iter().zip(&c.positive_units) {
            if !matches!(units, 1 | 2) || p.c.get(j).is_none_or(|v| !v.is_finite() || *v <= 0.) {
                return None;
            }
            if let Some(index) = positive.iter().position(|&old| old == j) {
                positive_units[index] = positive_units[index].checked_add(units)?;
                if positive_units[index] > 2 {
                    return None;
                }
            } else {
                positive.push(j);
                positive_units.push(units);
            }
        }
    }
    if positive.is_empty()
        || positive.len() > MAX_POSITIVE
        || negative
            .iter()
            .any(|&j| p.c.get(j).is_none_or(|v| !v.is_finite() || *v >= 0.))
    {
        return None;
    }
    let g: Vec<_> =
        a.g.iter()
            .zip(&b.g)
            .map(|(x, y)| finite(x + y))
            .collect::<Option<_>>()?;
    if g.len() != p.input_dim {
        return None;
    }
    let masks = 1usize.checked_shl(negative.len() as u32)?;
    let mut alpha = Vec::with_capacity(masks.checked_mul(positive.len())?);
    let left_masks = 1usize.checked_shl(a.negative.len() as u32)?;
    for mask in 0..masks {
        let ma = mask & (left_masks - 1);
        let mb = mask >> a.negative.len();
        for (&head, &units) in positive.iter().zip(&positive_units) {
            let mut numerator = 0.;
            for (c, m) in [(a, ma), (b, mb)] {
                if let Some(slot) = c.positive.iter().position(|&j| j == head) {
                    numerator = finite(
                        numerator
                            + f64::from(c.positive_units[slot])
                                * *c.alpha.get(m * c.positive.len() + slot)?,
                    )?;
                }
            }
            alpha.push(finite(numerator / f64::from(units))?.clamp(0., 1.));
        }
    }
    Some(Component {
        negative,
        positive,
        positive_units,
        g,
        alpha,
    })
}
/// Both checked forms own one common Component. Partial-only tail data stays
/// inline, with the original Vec allocations and serialized form tag intact.
#[derive(Debug)]
pub(super) struct Geometry {
    pub(super) body: Component,
    pub(super) partial: Option<partial::Details>,
}
impl Geometry {
    fn full(body: Component) -> Self {
        Self {
            body,
            partial: None,
        }
    }
    fn partial(witness: partial::Witness) -> Self {
        let (body, partial) = witness.into_parts();
        Self {
            body,
            partial: Some(partial),
        }
    }
    fn bits(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        let c = &self.body;
        match &self.partial {
            None => Box::new(
                [
                    0,
                    c.negative.len() as u64,
                    c.positive.len() as u64,
                    c.positive_units.len() as u64,
                    c.g.len() as u64,
                    c.alpha.len() as u64,
                ]
                .into_iter()
                .chain(c.negative.iter().chain(&c.positive).map(|v| *v as u64))
                .chain(c.positive_units.iter().map(|v| u64::from(*v)))
                .chain(c.g.iter().chain(&c.alpha).map(|v| v.to_bits())),
            ),
            Some(details) => Box::new([1].into_iter().chain(details.bits(c))),
        }
    }
    fn fields(&self) -> (&[usize], &[usize], &[u8], &[f64]) {
        let c = &self.body;
        let negative = self
            .partial
            .as_ref()
            .map_or(c.negative.as_slice(), |details| details.negative.as_slice());
        (negative, &c.positive, &c.positive_units, &c.g)
    }
}
#[derive(Debug)]
pub(super) struct PairPlane {
    pub(super) left: usize,
    pub(super) right: usize,
    pub(super) geometry: Geometry,
    pub(super) lower: f64,
}
#[derive(Debug)]
pub(super) struct Checked {
    pub(super) source: usize,
    pub(super) ledger: Arc<std::sync::Mutex<Counts>>,
    pub(super) witness: Arc<RetainedWords>,
    pub(super) signature: Arc<RetainedWords>,
    pub(super) clocks: (Instant, Instant),
    pub(super) planes: Vec<PairPlane>,
    pub(super) integrity: Vec<u64>,
}
impl Checked {
    fn bits(&self) -> impl Iterator<Item = u64> + '_ {
        [self.source as u64, self.planes.len() as u64]
            .into_iter()
            .chain(self.planes.iter().flat_map(|p| {
                [p.left as u64, p.right as u64, p.lower.to_bits()]
                    .into_iter()
                    .chain(p.geometry.bits())
            }))
    }
    fn finish(row: &SignedMajorant, planes: Vec<PairPlane>) -> Self {
        let mut out = Self {
            source: row.source,
            ledger: row.budget.state.clone(),
            witness: row.witness.clone(),
            signature: row.signature.clone(),
            clocks: (row.budget.authority, row.budget.search_end),
            planes,
            integrity: Vec::new(),
        };
        out.integrity = out.bits().collect();
        out
    }
    pub(super) fn owned(&self, row: &SignedMajorant) -> bool {
        self.source == row.source
            && self.source == row.budget.source
            && Arc::ptr_eq(&self.ledger, &row.budget.state)
            && Arc::ptr_eq(&self.witness, &row.witness)
            && Arc::ptr_eq(&self.signature, &row.signature)
            && self.clocks == (row.budget.authority, row.budget.search_end)
            && self.bits().eq(self.integrity.iter().copied())
    }
}
fn full(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    choice: &resources::Choice,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(Geometry, f64)> {
    let mut component = union(row, cuts.problem, choice)?;
    let mut coordinates = 0;
    let steered = steering::steer(cuts.problem, &mut component, end, poll, &mut coordinates);
    row.budget
        .state
        .lock()
        .ok()?
        .component_rows
        .joint
        .coordinates += coordinates;
    steered?;
    let mut lower = f64::INFINITY;
    require_hull_environment().ok()?;
    for mask in 0..choice.masks {
        if !live(end) || !poll() {
            return None;
        }
        lower = lower.min(finite(certificate::mask_lower(
            cuts, &component, mask, end, poll,
        )?)?);
        row.budget
            .state
            .lock()
            .ok()?
            .component_rows
            .joint
            .checked_masks += 1;
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some((Geometry::full(component), finite(lower)?))
}
pub(super) fn produce(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Checked> {
    let end = match &row.budget.prepaid {
        Some(ticket) => {
            let end = ticket.end(row, end)?;
            if !ticket.claim_checker(row, end) {
                return None;
            }
            end
        }
        None => end,
    };
    let choices = resources::choices(row, cuts, end, poll)?;
    {
        let mut c = row.budget.state.lock().ok()?;
        let r = &mut c.component_rows.joint;
        r.eligible_pairs = choices.len();
        r.partial_eligible_pairs = choices.iter().filter(|v| v.partial).count();
    }
    let mut selected = Vec::with_capacity(resources::MAX_PAIRS);
    let mut planes = Vec::with_capacity(resources::MAX_PAIRS);
    let limit = if row.budget.prepaid.is_some() {
        1
    } else {
        resources::MAX_PAIRS
    };
    for _ in 0..limit {
        if !live(end) || !poll() {
            stop(&row.budget, "deadline-or-poll");
            return None;
        }
        let best = if selected.is_empty() {
            resources::first_partial(row, &choices)
                .or_else(|| resources::pick(row, &choices, &selected))
        } else {
            resources::pick(row, &choices, &selected)
        };
        let Some(index) = best else {
            break;
        };
        if selected.is_empty() {
            let mut c = row.budget.state.lock().ok()?;
            if row.budget.prepaid.is_none() {
                if c.checks >= MAX_CHECKS {
                    c.component_rows.joint.stop_reason = "check-slots";
                    break;
                }
                c.checks += 1;
            }
            // The new policy already debited its one slot before refinement.
            c.component_rows.joint.checker_slots = 1;
        }
        let choice = &choices[index];
        resources::pay(row, choice.work, choice.bytes, end)?;
        selected.push(index);
        {
            let mut c = row.budget.state.lock().ok()?;
            let r = &mut c.component_rows.joint;
            r.paid_pairs += 1;
            r.quoted_masks = r.quoted_masks.checked_add(choice.masks)?;
            if choice.partial {
                r.partial_paid_pairs += 1;
                r.partial_work_units = r.partial_work_units.checked_add(choice.work)?;
            }
        }
        let (geometry, lower) = if choice.partial {
            let (w, k) = partial::produce(row, cuts, choice, end, poll)?;
            (Geometry::partial(w), k)
        } else {
            full(row, cuts, choice, end, poll)?
        };
        planes.push(PairPlane {
            left: choice.left,
            right: choice.right,
            geometry,
            lower: finite(lower)?,
        });
        let mut c = row.budget.state.lock().ok()?;
        let r = &mut c.component_rows.joint;
        r.completed_pairs += 1;
        if choice.partial {
            r.partial_completed_pairs += 1;
        }
    }
    (live(end) && poll()).then(|| Checked::finish(row, planes))
}
pub(super) fn export(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    checked: &Checked,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<RowSpec>> {
    let end = match &row.budget.prepaid {
        Some(ticket) => ticket.end(row, end)?,
        None => end,
    };
    if !binding.matches(cuts, parts) || !checked.owned(row) {
        return None;
    }
    let mut result = Vec::with_capacity(checked.planes.len());
    let mut heads = 0usize;
    let mut terms = 0usize;
    let mut partial_rows = 0usize;
    for plane in &checked.planes {
        if !live(end) || !poll() {
            return None;
        }
        let (negative, positive, units, g) = plane.geometry.fields();
        let mut coeffs = Vec::with_capacity(negative.len() + positive.len() + g.len());
        if plane.geometry.partial.is_some() {
            partial_rows += 1;
        }
        for &j in negative {
            coeffs.push((binding.head_column(j)?.0, finite(2. * cuts.problem.c[j])?));
            heads += 1;
        }
        for (&j, &units) in positive.iter().zip(units) {
            let v = match units {
                1 => cuts.problem.c[j],
                2 => finite(2. * cuts.problem.c[j])?,
                _ => return None,
            };
            coeffs.push((binding.head_column(j)?.0, v));
            heads += 1;
        }
        for (i, &g) in g.iter().enumerate() {
            let v = finite(-2. * g)?;
            if v != 0. {
                coeffs.push((parts.input_vars.get(i)?.0, v));
            }
        }
        terms = terms.checked_add(coeffs.len())?;
        result.push(RowSpec {
            lb: finite(2. * plane.lower)?,
            ub: f64::INFINITY,
            coeffs,
        });
    }
    require_hull_environment().ok()?;
    if !live(end) || !poll() {
        return None;
    }
    let mut counts = row.budget.state.lock().ok()?;
    let r = &mut counts.component_rows.joint;
    r.partial_prepared_rows = partial_rows;
    r.prepared_rows = result.len();
    r.prepared_head_terms = heads;
    r.prepared_terms = terms;
    if !result.is_empty() {
        r.stop_reason = "prepared";
    } else if r.stop_reason != "check-slots" {
        r.stop_reason = "no-affordable-pair";
    }
    Some(result)
}

#[cfg(test)]
mod geometry_layout_tests {
    use super::*;

    #[test]
    fn inline_geometry_keeps_both_serializations_and_the_original_size_bound() {
        const { assert!(size_of::<Geometry>() <= size_of::<partial::Witness>()) };
        let component = || Component {
            negative: vec![2],
            positive: vec![5],
            positive_units: vec![1],
            g: vec![-0.25],
            alpha: vec![0.75],
        };
        let full = Geometry::full(component());
        assert_eq!(
            full.bits().collect::<Vec<_>>(),
            vec![
                0,
                1,
                1,
                1,
                1,
                1,
                2,
                5,
                1,
                (-0.25f64).to_bits(),
                0.75f64.to_bits(),
            ]
        );
        let partial = Geometry::partial(partial::Witness {
            negative: vec![2, 3],
            body: component(),
            tail: vec![partial::Tail {
                head: 3,
                g: vec![0.5],
                upper: 2.,
            }],
            tail_constant: certificate::I { lo: -1., hi: 1. },
            tail_coeff: vec![certificate::I { lo: 3., hi: 4. }],
        });
        assert_eq!(
            partial.bits().collect::<Vec<_>>(),
            vec![
                1,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                (-1f64).to_bits(),
                1f64.to_bits(),
                2,
                3,
                2,
                5,
                1,
                (-0.25f64).to_bits(),
                0.75f64.to_bits(),
                3,
                1,
                2f64.to_bits(),
                0.5f64.to_bits(),
                3f64.to_bits(),
                4f64.to_bits(),
            ]
        );
        assert_eq!(full.fields(), (&[2][..], &[5][..], &[1][..], &[-0.25][..]));
        assert_eq!(
            partial.fields(),
            (&[2, 3][..], &[5][..], &[1][..], &[-0.25][..])
        );
    }
}
