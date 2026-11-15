// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Separately prepaid closed-cell supports; the original table stays intact.
use super::*;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_input_phase.rs"]
mod input_phase;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_input_phase_collect.rs"]
mod input_phase_collect;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_input_phase_resources.rs"]
pub(in crate::single_hidden_hull::mip_cuts::signed) mod input_phase_resources;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_joint.rs"]
pub(in crate::single_hidden_hull::mip_cuts::signed) mod joint;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_lp.rs"]
pub(in crate::single_hidden_hull::mip_cuts::signed) mod lp;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_numeric.rs"]
mod numeric;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_ray.rs"]
pub(in crate::single_hidden_hull::mip_cuts::signed) mod ray;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_support.rs"]
mod support;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_component_phase_cell_tests.rs"]
mod tests;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SignedPhaseCellReport {
    pub input_phase: input_phase_resources::Report,
    pub ray: ray::SignedPhaseRayReport,
    pub joint: joint::SignedPhaseJointReport,
    pub lp: lp::SignedTableSupportReport,
    pub attempted: bool,
    pub admitted: bool,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub component: Option<usize>,
    pub collected_masks: usize,
    pub improved_masks: usize,
    pub joint_improved_masks: usize,
    pub prepared_rows: usize,
    pub prepared_terms: usize,
    pub prepared_selectors: usize,
    pub appended_rows: usize,
}
// Fixed inline growth exists even when Cell is disabled or refused. Include
// simultaneous report/quote copies, optional table headers and state latches;
// admitted Cell payload below separately pays its vectors and numeric scratch.
pub(in crate::single_hidden_hull::mip_cuts::signed) const FIXED_METADATA_BYTES: usize = 4
    * size_of::<SignedPhaseCellReport>()
    + 4 * size_of::<Option<Quote>>()
    + 2 * size_of::<Option<Table>>()
    + 2 * size_of::<Option<lp::Support>>()
    + 7 * size_of::<usize>()
    + size_of::<[usize; 8]>()
    + 8 * align_of::<Collector>()
    // Fourth-family prepare counters, predecessor index, collector counter
    // and the always-present extra seal length word.
    + 7 * size_of::<usize>()
    // The automatic report/quote/table terms above include their growth.
    // Also pay the collector's candidate headers and rollback flag, the
    // admission's target header, two state latches, one seal length word,
    // and bounded transient move/selection metadata. Payload is quoted later.
    + 504;

/// Historical feature-isolation controls discard only the paid Cell capability.
/// Keep its debt and every older capability, clock and once latch intact.
#[cfg(test)]
pub(in crate::single_hidden_hull::mip_cuts::signed) fn omit_after_payment_for_test(b: &Budget) {
    let mut counts = b.state.lock().unwrap();
    assert_eq!(counts.replay_attempts, 0);
    assert!(!counts.phase_cube.cell_attempted);
    let admission = counts.phase_cube.admission.as_mut().unwrap();
    assert!(admission.quote.cell.take().is_some());
}
#[derive(Debug, Clone, Copy)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Quote {
    pub input_phase: Option<input_phase_resources::Quote>,
    pub ray: Option<ray::Quote>,
    pub lp: Option<lp::Quote>,
    pub joint: Option<joint::Quote>,
    pub k: usize,
    masks: usize,
    pub terms: usize,
    pub setup: usize,
    pub work: usize,
    pub bytes: usize,
}
impl Quote {
    pub(super) fn seal_words(self) -> usize {
        self.masks
            + 12
            + self.ray.map_or(0, |_| self.masks)
            + 1
            + self.joint.map_or(0, |_| self.masks)
            + 1
            + self.input_phase.map_or(0, |q| q.d)
    }
}
pub(in crate::single_hidden_hull::mip_cuts::signed) fn offer(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    protected: usize,
) -> Option<Quote> {
    if !b.portfolio_phase_enabled
        || b.source != std::ptr::from_ref(cuts.problem) as usize
        || !live(b.search_end)
    {
        return None;
    }
    {
        let mut c = b.state.lock().ok()?;
        if c.phase_cube.cell_offered || c.phase_cube.admission.is_some() || c.replay_attempts != 0 {
            return None;
        }
        c.phase_cube.cell_offered = true;
        report(&mut c).cell.attempted = true;
    }
    let (d, h, _) = b.dimensions;
    let setup = 32usize
        .checked_mul(
            h.checked_add(d)?
                .checked_add(plan.components.len())?
                .checked_add(2)?,
        )?
        .checked_add(4096)?;
    {
        let mut c = b.state.lock().ok()?;
        // The pending All+Cube hold has priority over this new optional setup.
        // Refusing before its metadata walk spends nothing and preserves Cube.
        let next = c.work.checked_add(setup)?;
        if next
            .checked_add(c.reserved)?
            .checked_add(c.component_reserved)?
            .checked_add(protected)?
            > MAX_WORK
            || !live(b.search_end)
        {
            return None;
        }
        c.work = next;
        c.component_rows.work_units = c.component_rows.work_units.checked_add(setup)?;
        c.component_rows.phase.work_units = c.component_rows.phase.work_units.checked_add(setup)?;
        let r = report(&mut c);
        r.work_units = r.work_units.checked_add(setup)?;
        r.cell.work_units = setup;
    }
    let k = plan.components.iter().map(|c| c.negative.len()).max()?;
    if !(1..=8).contains(&k)
        || size_of::<Collector>() > 1280
        || size_of::<Table>() > 192
        || size_of::<Admission>() > 512
        || size_of::<Option<input_phase_collect::State>>() > 192
    {
        return None;
    }
    let masks = 1usize << k;
    let visits = k.checked_mul(masks)?;
    let terms = d.checked_add(32)?;
    // Across the complete cube exactly half of each bit's masks are inactive.
    // The second candidate changes one positive and one negative coefficient.
    // Includes pair setup, K candidate selection, copies and both box prices.
    let rays = 32usize
        .checked_mul(d.checked_add(1)?)?
        .checked_mul(
            (visits / 2)
                .checked_add(6usize.checked_mul(masks)?)?
                .checked_add(MAX_POSITIVE.checked_mul(k)?)?,
        )?
        .checked_add(16usize.checked_mul(visits)?)?;
    let support = 64usize
        .checked_mul(visits)?
        .checked_add(32usize.checked_mul(masks)?)?
        .checked_add(64usize.checked_mul((k + 1).checked_pow(2)?)?)?
        .checked_add(
            128usize.checked_mul(
                terms
                    .checked_add(2 * k)?
                    .checked_add(h)?
                    .checked_add(d)?
                    .checked_add(2)?,
            )?,
        )?
        .checked_add(4096)?;
    // One value vector and one integrity copy; no table clones.
    let bytes = 16usize
        .checked_mul(masks)?
        .checked_add(64usize.checked_mul(terms)?)?
        .checked_add(128usize.checked_mul(k)?)?
        .checked_add(768)?
        .checked_add(64usize.checked_mul(d.checked_add(1)?)?)?
        .checked_add(16usize.checked_mul(d)?)?
        .checked_add(4096)?;
    Some(Quote {
        input_phase: None,
        ray: None,
        lp: None,
        joint: None,
        k,
        masks,
        terms,
        setup,
        work: setup.checked_add(rays)?.checked_add(support)?,
        bytes,
    })
}

#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts) struct Table {
    pub(super) component: usize,
    values: Vec<f64>,
    activity: Option<[f64; 8]>,
    ray: Option<Vec<f64>>,
    joint: Option<Vec<f64>>,
    input_g: Option<Vec<f64>>,
}
impl Table {
    /// Borrow the sealed slope only for the complete Cube producer's native
    /// row descriptor. Publication copies the emitted row's exact terms.
    pub(super) fn native_input_g(&self) -> Option<&[f64]> {
        self.input_g.as_deref()
    }

    pub(super) fn bits(&self) -> impl Iterator<Item = u64> + '_ {
        [
            self.component as u64,
            self.values.len() as u64,
            u64::from(self.activity.is_some()),
        ]
        .into_iter()
        .chain(self.values.iter().map(|x| x.to_bits()))
        .chain(self.activity.iter().flatten().map(|x| x.to_bits()))
        .chain(std::iter::once(self.ray.as_ref().map_or(0, Vec::len) as u64))
        .chain(self.ray.iter().flatten().map(|x| x.to_bits()))
        .chain(std::iter::once(
            self.joint.as_ref().map_or(0, Vec::len) as u64
        ))
        .chain(self.joint.iter().flatten().map(|x| x.to_bits()))
        .chain(std::iter::once(
            self.input_g.as_ref().map_or(0, Vec::len) as u64
        ))
        .chain(self.input_g.iter().flatten().map(|x| x.to_bits()))
    }
    pub(super) fn valid(&self, q: Quote, base: &[Vec<f64>], plan: &Plan) -> bool {
        let Some(c) = plan.components.get(self.component) else {
            return false;
        };
        let Some(old) = base.get(self.component) else {
            return false;
        };
        c.negative.len() == q.k
            && self.values.len() == q.masks
            && old.len() == q.masks
            && self.values.capacity() <= q.masks
            && self
                .values
                .iter()
                .zip(old)
                .all(|(a, b)| a.is_finite() && *a >= *b)
            && self
                .activity
                .iter()
                .flatten()
                .all(|a| a.is_finite() && (0. ..=1.).contains(a))
            && match (&self.ray, q.ray) {
                (None, None) => true,
                (Some(values), Some(_)) => {
                    values.len() == q.masks
                        && values.capacity() <= q.masks
                        && values
                            .iter()
                            .zip(&self.values)
                            .all(|(a, b)| a.is_finite() && *a >= *b)
                }
                _ => false,
            }
            && match (&self.joint, q.joint, &self.ray) {
                (None, None, _) => true,
                (Some(values), Some(_), Some(ray)) => {
                    values.len() == q.masks
                        && values.capacity() <= q.masks
                        && match (&self.input_g, q.input_phase) {
                            (None, _) => values
                                .iter()
                                .zip(ray)
                                .all(|(a, b)| a.is_finite() && *a >= *b),
                            (Some(g), Some(input)) => {
                                input.d == c.g.len()
                                    && g.len() == input.d
                                    && g.capacity() <= input.d
                                    && g.iter().all(|v| v.is_finite())
                                    && values.iter().all(|v| v.is_finite())
                            }
                            _ => false,
                        }
                }
                _ => false,
            }
            && (self.input_g.is_none() || self.joint.is_some())
    }
}
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Collector {
    quote: Quote,
    table: Table,
    pairs: [[Option<numeric::Pair>; 8]; 2],
    improved: usize,
    joint_improved: usize,
    ray_improved: usize,
    block_improved: usize,
    input_phase: Option<input_phase_collect::State>,
    input_phase_failed: bool,
}
/// An optional Cell refusal leaves the ordinary relaxation available.
/// The outer Option remains the terminal cancellation path.
pub(in crate::single_hidden_hull::mip_cuts::signed) enum CellAttempt<T> {
    Refused,
    Ready(T),
}
impl<T> CellAttempt<T> {
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn into_option(self) -> Option<T> {
        match self {
            Self::Refused => None,
            Self::Ready(value) => Some(value),
        }
    }
}
/// None means terminal cancellation; Refused retains the ordinary path.
pub(in crate::single_hidden_hull::mip_cuts::signed) fn begin(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<CellAttempt<Collector>> {
    if !live(end) || !poll() {
        return None;
    }
    let chosen = {
        let mut counts = b.state.lock().ok()?;
        let Some(a) = counts.phase_cube.admission.as_ref() else {
            return Some(CellAttempt::Refused);
        };
        let Some(q) = a.quote.cell else {
            return Some(CellAttempt::Refused);
        };
        if !owned(a, b)
            || counts.phase_cube.cell_attempted
            || counts.replay_attempts != 1
            || counts.source_replays != 0
            || report_ref(&counts).collection_attempts != 1
            || report_ref(&counts).reserved_work_units != 0
        {
            return Some(CellAttempt::Refused);
        }
        // Claim before the bounded metadata walk. Callbacks may themselves
        // inspect this ledger, so none may run while its mutex is held.
        counts.phase_cube.cell_attempted = true;
        let a = counts.phase_cube.admission.as_ref()?;
        let mut best = None;
        for (index, item) in plan.components.iter().enumerate() {
            if !live(end) {
                return None;
            }
            if item.negative.len() != q.k {
                continue;
            }
            let mut values = [0.; 8];
            let mut have = true;
            let mut fractionality = 0.;
            for (bit, &head) in item.negative.iter().enumerate() {
                match a
                    .activity
                    .as_ref()
                    .and_then(|v| v.get(head))
                    .copied()
                    .flatten()
                {
                    Some(v) if v.is_finite() && (0. ..=1.).contains(&v) => {
                        values[bit] = v;
                        fractionality += v.min(1. - v);
                    }
                    _ => have = false,
                }
            }
            let score = if have { 1. + fractionality } else { 0. };
            if best.as_ref().is_none_or(|(_, _, old)| score > *old) {
                best = Some((index, have.then_some(values), score));
            }
        }
        let target = counts.phase_cube.admission.as_mut()?.input_target.take();
        best.map(|(index, activity, _)| (q, index, activity, target))
    };
    if !live(end) || !poll() {
        return None;
    }
    let Some((quote, index, activity, target)) = chosen else {
        return Some(CellAttempt::Refused);
    };
    let pairs = numeric::pairs(cuts.problem, &plan.components[index], end, poll)?;
    let values = Vec::with_capacity(quote.masks);
    let ray = quote.ray.map(|_| Vec::with_capacity(quote.masks));
    let mut joint = quote.joint.map(|_| Vec::with_capacity(quote.masks));
    if values.capacity() > quote.masks
        || ray.as_ref().is_some_and(|v| v.capacity() > quote.masks)
        || joint.as_ref().is_some_and(|v| v.capacity() > quote.masks)
    {
        return Some(CellAttempt::Refused);
    }
    let input_phase = match (quote.input_phase, target, joint.as_mut(), activity) {
        (Some(input), Some(target), Some(scratch), Some(_))
            if input.d == cuts.problem.input_dim =>
        {
            input_phase_collect::State::begin(
                cuts.problem,
                &plan.components[index],
                target,
                scratch,
                end,
                poll,
            )?
            .into_option()
        }
        _ => None,
    };
    Some(CellAttempt::Ready(Collector {
        quote,
        table: Table {
            component: index,
            values,
            activity,
            ray,
            joint,
            input_g: None,
        },
        pairs,
        improved: 0,
        joint_improved: 0,
        ray_improved: 0,
        block_improved: 0,
        input_phase,
        input_phase_failed: false,
    }))
}
fn report_ref(c: &Counts) -> &SignedPhaseCubeReport {
    &c.component_rows.phase.cube
}
impl Collector {
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn apply(
        &mut self,
        cuts: &HullRootMipCuts<'_>,
        c: &Component,
        index: usize,
        mask: usize,
        coefficients: &[certificate::I],
        constant: certificate::I,
        base: f64,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        self.apply_with_advice(
            cuts,
            c,
            index,
            mask,
            coefficients,
            constant,
            base,
            end,
            poll,
        )
        .map(|_| ())
    }
    /// Preserve the original result while exporting already-paid Ray advice.
    /// The receiver must validate and independently reprice it from Source.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn apply_with_advice(
        &mut self,
        cuts: &HullRootMipCuts<'_>,
        c: &Component,
        index: usize,
        mask: usize,
        coefficients: &[certificate::I],
        constant: certificate::I,
        base: f64,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<(bool, Option<(usize, f64)>)> {
        if index != self.table.component {
            return Some((false, None));
        }
        let mut captured = (false, None);
        if mask != self.table.values.len()
            || mask >= self.quote.masks
            || c.negative.len() != self.quote.k
        {
            return None;
        }
        let (value, joint) = numeric::strengthen(
            cuts.problem,
            c,
            mask,
            coefficients,
            constant,
            base,
            &self.pairs,
            end,
            poll,
        )?;
        self.improved += usize::from(value > base);
        self.joint_improved += usize::from(joint);
        self.table.values.push(value);
        if let Some(values) = &mut self.table.ray {
            let (stronger, advice) = ray::strengthen_with_advice(
                cuts.problem,
                c,
                mask,
                coefficients,
                constant,
                value,
                end,
                poll,
            )?;
            captured = (true, advice);
            self.ray_improved += usize::from(stronger > value);
            values.push(stronger);
            if let Some(joint) = &mut self.table.joint {
                let block = joint::strengthen(
                    cuts.problem,
                    c,
                    mask,
                    coefficients,
                    constant,
                    stronger,
                    &self.pairs,
                    end,
                    poll,
                )?;
                self.block_improved += usize::from(block > stronger);
                let next = if let Some(input) = &mut self.input_phase {
                    if block.to_bits() != stronger.to_bits() {
                        None
                    } else {
                        input
                            .prices(
                                cuts.problem,
                                c,
                                mask,
                                coefficients,
                                constant,
                                advice,
                                end,
                                poll,
                            )?
                            .into_option()
                    }
                } else {
                    Some(block)
                };
                if let Some(next) = next {
                    joint.push(next);
                } else {
                    // Every earlier legacy Joint word equalled Ray. Recover
                    // that exact prefix before retaining this and later words;
                    // the paid candidate work/storage is never refunded.
                    joint.copy_from_slice(&values[..mask]);
                    self.input_phase = None;
                    self.input_phase_failed = true;
                    joint.push(block);
                }
            }
        }
        Some(captured)
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn finish(
        mut self,
        plan: &Plan,
        b: &Budget,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Table> {
        if self.table.values.len() != self.quote.masks
            || self
                .table
                .ray
                .as_ref()
                .is_some_and(|v| v.len() != self.quote.masks)
            || self
                .table
                .joint
                .as_ref()
                .is_some_and(|v| v.len() != self.quote.masks)
        {
            return None;
        }
        let mut selected = None;
        let candidates = usize::from(self.input_phase.is_some()) * 4;
        if let Some(input) = self.input_phase.take() {
            let c = plan.components.get(self.table.component)?;
            let result = input.finish(
                c,
                self.table.joint.take()?,
                self.table.ray.as_deref()?,
                &self.table.activity?[..self.quote.k],
                end,
                poll,
            )?;
            selected = result.candidate;
            self.table.joint = Some(result.values);
            self.table.input_g = result.g;
        }
        let mut counts = b.state.lock().ok()?;
        let r = &mut report(&mut counts).cell;
        r.component = Some(self.table.component);
        r.collected_masks = self.table.values.len();
        r.improved_masks = self.improved;
        r.joint_improved_masks = self.joint_improved;
        r.ray.masks = self.table.ray.as_ref().map_or(0, Vec::len);
        r.ray.improved_masks = self.ray_improved;
        r.joint.masks = self.table.joint.as_ref().map_or(0, Vec::len);
        r.joint.improved_masks = self.block_improved;
        if self.quote.input_phase.is_some() {
            r.input_phase.masks = if candidates == 0 { 0 } else { self.quote.masks };
            r.input_phase.candidates = candidates;
            r.input_phase.selected_candidate = selected;
            r.input_phase.stop_reason = if self.input_phase_failed {
                "legacy-or-numeric-rollback"
            } else if selected.is_some() {
                "selected"
            } else if candidates > 0 {
                "no-checked-gain"
            } else if r.input_phase.stop_reason == "capture-refused" {
                "capture-refused"
            } else {
                "no-proposal"
            };
        }
        Some(self.table)
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Family {
    Cell,
    Ray,
    Lp,
    Joint,
}

pub(super) fn prepare(
    table: &Table,
    q: Quote,
    row: &SignedMajorant,
    binding: &binding::SignedPhaseBinding<'_>,
    base: &RowSpec,
    family: Family,
    incumbent: &mut Option<lp::Support>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<CellAttempt<(RowSpec, usize)>> {
    let c = row.plan.components.get(table.component)?;
    let input_g = if family == Family::Joint {
        table.input_g.as_deref()
    } else {
        None
    };
    if input_g.is_some() {
        let mut counts = row.budget.state.lock().ok()?;
        if counts.phase_cube.input_phase_prepared {
            return Some(CellAttempt::Refused);
        }
        counts.phase_cube.input_phase_prepared = true;
    }
    let values = match family {
        Family::Joint => {
            if q.joint.is_none() {
                return Some(CellAttempt::Refused);
            }
            table.joint.as_ref()?
        }
        Family::Lp => return Some(CellAttempt::Refused),
        Family::Cell => &table.values,
        Family::Ray => {
            if q.ray.is_none() {
                return Some(CellAttempt::Refused);
            }
            table.ray.as_ref()?
        }
    };
    let weights = support_weights(
        values,
        &c.negative,
        table.activity.as_ref().map(|a| &a[..q.k]),
    );
    let Some(weights) = weights else {
        return Some(CellAttempt::Refused);
    };
    let (mut encoded, mut columns, mut selectors) = ([0.; 8], [None; 8], 0usize);
    for (bit, &head) in c.negative.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        columns[bit] = binding.column(head);
        if columns[bit].is_some() {
            let Some(value) = finite(-2. * weights[bit]) else {
                return Some(CellAttempt::Refused);
            };
            encoded[bit] = value;
            selectors += usize::from(value != 0.);
        }
    }
    let mut cancelled = false;
    let lower = repair(values, &encoded[..q.k], end, &mut || {
        let ok = poll();
        cancelled |= !ok;
        ok
    });
    if cancelled || !live(end) {
        return None;
    }
    let Some(mut lower) = lower else {
        return Some(CellAttempt::Refused);
    };
    if family == Family::Ray || family == Family::Joint {
        let mapped = columns.map(|column| column.is_some());
        (encoded, lower) = support::choose(
            values,
            encoded,
            lower,
            &mapped[..q.k],
            table.activity.as_ref().map(|a| &a[..q.k]),
            end,
            poll,
        )?;
        selectors = encoded[..q.k].iter().filter(|v| **v != 0.).count();
        if family == Family::Ray {
            *incumbent = Some(lp::Support { a: encoded, lower });
        }
    }
    if input_g.is_none() && selectors == 0 && lower <= base.lb {
        return Some(CellAttempt::Refused);
    }
    let head_terms = c.negative.len().checked_add(c.positive.len())?;
    let needed = if let Some(g) = input_g {
        if g.len() != c.g.len() || q.input_phase.is_none_or(|quote| quote.d != g.len()) {
            return Some(CellAttempt::Refused);
        }
        head_terms.checked_add(g.len())?.checked_add(q.k)?
    } else {
        base.coeffs.len().checked_add(q.k)?
    };
    if needed > q.terms {
        return Some(CellAttempt::Refused);
    }
    let mut coeffs = Vec::with_capacity(needed);
    if coeffs.capacity() > q.terms {
        return Some(CellAttempt::Refused);
    }
    if let Some(g) = input_g {
        // The caller checked all::base_valid against the original model.
        // Retain those exact original head coefficients, then REBUILD every
        // input term from stored g. Adding a rounded delta to the old sparse
        // row would fail to authenticate its actual serialized coefficient.
        coeffs.extend_from_slice(base.coeffs.get(..head_terms)?);
        for (input, &g) in g.iter().enumerate() {
            if !live(end) || !poll() {
                return None;
            }
            let Some(value) = finite(-2. * g) else {
                return Some(CellAttempt::Refused);
            };
            if value != 0. {
                coeffs.push((binding.input_column(input)?.0, value));
            }
        }
    } else {
        coeffs.extend_from_slice(&base.coeffs);
    }
    for bit in 0..q.k {
        if encoded[bit] != 0. {
            coeffs.push((columns[bit]?.0, encoded[bit]));
        }
    }
    Some(CellAttempt::Ready((
        RowSpec {
            lb: lower,
            ub: f64::INFINITY,
            coeffs,
        },
        selectors,
    )))
}
